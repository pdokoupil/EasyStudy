"""RecBole (PyTorch) algorithm wrapper for fastcompare.

Gives EasyStudy access to RecBole's well-maintained model zoo (BPR, LightGCN, NeuMF, …) as
an alternative to the TensorFlow-Recommenders backend. Requires the optional extra::

    pip install "easystudy[recbole]"        # torch + recbole

Because torch/recbole are imported at module top, fastcompare's resilient plugin discovery
simply **skips** this module when the extra isn't installed (exactly like the `vae` plugin
with Microsoft `recommenders`), so the lightweight core is unaffected.

Cold-user prediction: RecBole's general recommenders learn user/item embeddings on a fixed
training set and don't natively score *unseen* users. fastcompare needs exactly that (a new
participant described only by their preference-elicitation selections). We therefore build a
pseudo-user profile as the mean of the learned embeddings of the selected items and rank all
items by similarity to it — a standard, model-agnostic cold-start approximation for
embedding-based recommenders.

⚠️ EXPERIMENTAL: this wrapper compiles and is skipped safely without the extra, but the
training/prediction path needs a live smoke test with torch+recbole installed (RecBole's
data/config APIs vary across versions).
"""
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer

from plugins.fastcompare.algo.algorithm_base import (
    AlgorithmBase,
    Parameter,
    ParameterType,
)

# Embedding-based general recommenders that expose item embeddings we can use for the
# cold-user approximation below.
SUPPORTED_MODELS = ["BPR", "LightGCN", "NeuMF", "NGCF", "DMF"]


class RecBole(AlgorithmBase):
    """Train a RecBole general recommender and serve cold-user recommendations."""

    def __init__(self, loader, model="BPR", epochs=50, embedding_size=64, **kwargs):
        self._loader = loader
        self._ratings_df = loader.ratings_df
        self._all_items = self._ratings_df.item.unique()
        self._model_name = model
        self._epochs = int(epochs)
        self._embedding_size = int(embedding_size)

        self._model = None
        self._dataset = None

    # --- training -----------------------------------------------------------------------
    def fit(self):
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

        config = Config(model=self._model_name, dataset=name, config_dict={
            "data_path": tmpdir,
            "USER_ID_FIELD": "user_id",
            "ITEM_ID_FIELD": "item_id",
            "RATING_FIELD": "rating",
            "load_col": {"inter": ["user_id", "item_id", "rating"]},
            "epochs": self._epochs,
            "embedding_size": self._embedding_size,
            "eval_args": {"split": {"RS": [1.0, 0.0, 0.0]}, "order": "RO",
                          "group_by": "user", "mode": "full"},
            "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1,
                                      "alpha": 1.0, "dynamic": False, "candidate_num": 0},
            "device": "cpu",
            "checkpoint_dir": os.path.join(tmpdir, "ckpt"),
            "save_dataset": False,
            "save_dataloaders": False,
            "show_progress": False,
        })
        init_seed(config["seed"], config["reproducibility"])

        dataset = create_dataset(config)
        train_data, valid_data, _ = data_preparation(config, dataset)

        train_dataset = getattr(train_data, "dataset", None) or getattr(train_data, "_dataset")
        model = get_model(config["model"])(config, train_dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(train_data, valid_data=None, saved=False, show_progress=False)

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
            f"RecBole model '{self._model_name}' does not expose item embeddings for "
            f"cold-user prediction; try one of {SUPPORTED_MODELS}."
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

        # rank candidates by the pseudo-user score, map internal id back to loader index
        scored = []
        for cand in candidates:
            internal = self._loader_index_to_internal(cand)
            if internal is not None and internal > 0:
                scored.append((scores[internal], int(cand)))
        scored.sort(reverse=True)
        return [item for _, item in scored[:k]]

    @classmethod
    def name(cls):
        return "RecBole (PyTorch)"

    @classmethod
    def parameters(cls):
        return [
            Parameter("model", ParameterType.OPTIONS, "BPR",
                      help=f"RecBole model to train. Supported for cold-user prediction: "
                           f"{', '.join(SUPPORTED_MODELS)}.",
                      options=SUPPORTED_MODELS),
            Parameter("epochs", ParameterType.INT, 50,
                      help="Number of training epochs."),
            Parameter("embedding_size", ParameterType.INT, 64,
                      help="Embedding dimensionality."),
        ]
