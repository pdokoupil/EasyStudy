"""RecBole (PyTorch) algorithms for fastcompare — one AlgorithmBase subclass per model.

Each supported RecBole model is its own `AlgorithmBase` subclass (BPR, LightGCN, NeuMF, NGCF,
DMF) so it exposes only *its own* hyperparameters in the study-creation UI. Shared training /
cold-user-prediction logic lives in the `_RecBoleAlgorithm` mixin (not an AlgorithmBase, so
discovery doesn't pick it up on its own).

Requires the optional extra::

    pip install "easystudy[recbole]"        # torch + recbole

torch/recbole are imported at module top, so fastcompare's resilient discovery **skips** this
module when the extra isn't installed (the lightweight core is unaffected).

Cold-user prediction: RecBole's general recommenders learn user/item embeddings on a fixed
training set and don't natively score *unseen* users. fastcompare needs exactly that (a new
participant described only by their preference-elicitation selections), so we build a
pseudo-user profile as the mean of the selected items' learned embeddings and rank all items
by similarity to it — a standard, model-agnostic cold-start approximation.
"""
import os
import shutil
import tempfile

import numpy as np
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)


def _patch_scipy_dok():
    """Restore ``scipy.sparse.dok_matrix._update``, removed in modern scipy.

    RecBole's graph models (LightGCN, NGCF, GCMC, …) build the normalized adjacency matrix
    with ``A._update(data_dict)`` — an internal that newer scipy dropped, causing
    ``AttributeError: 'dok_matrix' object has no attribute '_update'`` (RecBole PR #2187,
    unreleased in 1.2.1). We re-add a compatible implementation via the public setter.
    """
    try:
        from scipy.sparse import dok_matrix
    except Exception:
        return
    if hasattr(dok_matrix, "_update"):
        return

    def _update(self, data):  # data: mapping {(row, col): value}
        for key, value in data.items():
            self[key] = value

    dok_matrix._update = _update


def _coerce(value):
    """Best-effort numeric coercion so UI-provided strings become int/float for RecBole."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


class _RecBoleAlgorithm:
    """Shared training + cold-user prediction for RecBole models.

    Subclasses set ``MODEL`` (the RecBole model name) and implement ``name`` / ``parameters``.
    Any parameter the subclass declares is forwarded verbatim into RecBole's config, so the
    parameter *names* must be valid RecBole config keys (e.g. ``embedding_size``, ``n_layers``).
    """

    #: RecBole model name, e.g. "BPR". Set by each concrete subclass.
    MODEL: str = None

    def __init__(self, loader, **kwargs):
        self._loader = loader
        self._ratings_df = loader.ratings_df
        self._all_items = self._ratings_df.item.unique()
        # everything the study-creation UI passed for this model -> RecBole config overrides
        self._config_overrides = {k: _coerce(v) for k, v in kwargs.items()}
        self._model = None
        self._dataset = None

    # --- training -----------------------------------------------------------------------
    def fit(self):
        _patch_scipy_dok()  # must run before RecBole builds any graph adjacency matrix

        df = self._ratings_df[["user", "item"]].dropna().astype({"user": int, "item": int})
        tmpdir = tempfile.mkdtemp(prefix="easystudy_recbole_")
        name = "easystudy"
        dsdir = os.path.join(tmpdir, name)
        os.makedirs(dsdir, exist_ok=True)

        # RecBole reads "atomic" TSV files; item tokens are the loader's 0-based item indices.
        inter = df.copy()
        inter.columns = ["user_id:token", "item_id:token"]
        inter["rating:float"] = 1.0
        inter.to_csv(os.path.join(dsdir, f"{name}.inter"), sep="\t", index=False)

        config_dict = {
            "data_path": tmpdir,
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "load_col": {"inter": ["user_id", "item_id", "rating"]},
            "eval_args": {"split": {"RS": [1.0, 0.0, 0.0]}, "order": "RO",
                          "group_by": "user", "mode": "full"},
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1,
                                      "alpha": 1.0, "dynamic": False, "candidate_num": 0},
            "device": "cpu",
            "checkpoint_dir": os.path.join(tmpdir, "ckpt"),
            "save_dataset": False,
            "save_dataloaders": False,
            "show_progress": False,
            "epochs": 30,               # sensible default; overridden below if the model exposes it
            **self._config_overrides,   # model-specific hyperparameters from the UI
        }
        config = Config(model=self.MODEL, dataset=name, config_dict=config_dict)
        init_seed(config["seed"], config["reproducibility"])

        # RecBole writes several artifacts (log/, log_tensorboard/, saved/) to *relative* paths
        # under the process's current working directory, regardless of `checkpoint_dir` — so
        # without this they'd pollute wherever the app happens to be run from. Redirect into our
        # own tmpdir (which we remove afterward) instead.
        prev_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            dataset = create_dataset(config)
            train_data, _valid_data, _test_data = data_preparation(config, dataset)

            train_dataset = getattr(train_data, "dataset", None) or getattr(train_data, "_dataset")
            model = get_model(config["model"])(config, train_dataset).to(config["device"])
            trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
            trainer.fit(train_data, valid_data=None, saved=False, show_progress=False)
        finally:
            os.chdir(prev_cwd)
            shutil.rmtree(tmpdir, ignore_errors=True)

        self._model = model.eval()
        self._dataset = dataset

    # --- prediction ---------------------------------------------------------------------
    def _item_embeddings(self):
        """Best-effort extraction of the learned item-embedding matrix as a tensor."""
        model = self._model
        if hasattr(model, "item_embedding") and hasattr(model.item_embedding, "weight"):
            return model.item_embedding.weight.detach()
        # LightGCN / NGCF compute propagated embeddings in forward() -> (user_all, item_all)
        try:
            out = model.forward()
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return out[1].detach()
        except Exception:
            pass
        raise RuntimeError(
            f"RecBole model '{self.MODEL}' does not expose item embeddings for cold-user "
            f"prediction; use an embedding-based model (BPR, LightGCN, NeuMF, NGCF, DMF)."
        )

    def _loader_index_to_internal(self, item_index):
        # tokens were written as str(loader_index); RecBole maps token -> internal id.
        try:
            return int(self._dataset.token2id("item_id", str(int(item_index))))
        except Exception:
            return None

    def predict(self, selected_items, filter_out_items, k):
        candidates = np.setdiff1d(self._all_items, np.asarray(selected_items, dtype=int))
        candidates = np.setdiff1d(candidates, np.asarray(filter_out_items, dtype=int))

        sel_internal = [i for i in (self._loader_index_to_internal(s) for s in selected_items)
                        if i is not None and i > 0]
        if not sel_internal:
            # unseen user with no usable history -> random (mirrors EASE's fallback)
            return np.random.choice(candidates, size=min(k, len(candidates)), replace=False).tolist()

        with torch.no_grad():
            item_emb = self._item_embeddings()                       # [n_internal, dim]
            profile = item_emb[sel_internal].mean(dim=0)             # [dim]
            scores = (item_emb @ profile).cpu().numpy()             # [n_internal]

        scored = []
        for cand in candidates:
            internal = self._loader_index_to_internal(cand)
            if internal is not None and internal > 0:
                scored.append((scores[internal], int(cand)))
        scored.sort(reverse=True)
        return [item for _, item in scored[:k]]


# --- concrete, discoverable per-model algorithms ----------------------------------------
# Parameter names must be valid RecBole config keys (they're forwarded straight to RecBole).

def _epochs(default=30):
    return Parameter("epochs", ParameterType.INT, default, help="Number of training epochs.")


def _learning_rate(default=0.001):
    return Parameter("learning_rate", ParameterType.FLOAT, default, help="Optimizer learning rate.")


class BPR(_RecBoleAlgorithm, AlgorithmBase):
    """Bayesian Personalized Ranking (matrix factorization)."""
    MODEL = "BPR"

    @classmethod
    def name(cls):
        return "BPR (RecBole)"

    @classmethod
    def parameters(cls):
        return [Parameter("embedding_size", ParameterType.INT, 64, help="Embedding dimensionality."),
                _epochs(), _learning_rate()]


class LightGCN(_RecBoleAlgorithm, AlgorithmBase):
    """LightGCN — graph-convolution collaborative filtering."""
    MODEL = "LightGCN"

    @classmethod
    def name(cls):
        return "LightGCN (RecBole)"

    @classmethod
    def parameters(cls):
        return [Parameter("embedding_size", ParameterType.INT, 64, help="Embedding dimensionality."),
                Parameter("n_layers", ParameterType.INT, 2, help="Number of graph-convolution layers."),
                Parameter("reg_weight", ParameterType.FLOAT, 1e-05, help="L2 regularization weight."),
                _epochs(), _learning_rate()]


class NGCF(_RecBoleAlgorithm, AlgorithmBase):
    """Neural Graph Collaborative Filtering."""
    MODEL = "NGCF"

    @classmethod
    def name(cls):
        return "NGCF (RecBole)"

    @classmethod
    def parameters(cls):
        return [Parameter("embedding_size", ParameterType.INT, 64, help="Embedding dimensionality."),
                Parameter("reg_weight", ParameterType.FLOAT, 1e-05, help="L2 regularization weight."),
                Parameter("node_dropout", ParameterType.FLOAT, 0.0, help="Node dropout rate."),
                Parameter("message_dropout", ParameterType.FLOAT, 0.1, help="Message dropout rate."),
                _epochs(), _learning_rate()]


class NeuMF(_RecBoleAlgorithm, AlgorithmBase):
    """Neural Matrix Factorization (NCF)."""
    MODEL = "NeuMF"

    @classmethod
    def name(cls):
        return "NeuMF (RecBole)"

    @classmethod
    def parameters(cls):
        return [Parameter("mf_embedding_size", ParameterType.INT, 64, help="GMF embedding size."),
                Parameter("mlp_embedding_size", ParameterType.INT, 64, help="MLP embedding size."),
                _epochs(), _learning_rate()]


class DMF(_RecBoleAlgorithm, AlgorithmBase):
    """Deep Matrix Factorization."""
    MODEL = "DMF"

    @classmethod
    def name(cls):
        return "DMF (RecBole)"

    @classmethod
    def parameters(cls):
        return [Parameter("user_embedding_size", ParameterType.INT, 64, help="User embedding size."),
                Parameter("item_embedding_size", ParameterType.INT, 64, help="Item embedding size."),
                _epochs(), _learning_rate()]
