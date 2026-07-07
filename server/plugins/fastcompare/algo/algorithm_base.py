from abc import ABC, abstractmethod
from typing import Any
import pickle


class ParameterType:
    """Types of hyperparameters / configurable parameters exposed by a component.

    Used as the ``param_type`` of a `Parameter`. ``OPTIONS`` lets the user choose
    one out of several predefined options.
    """
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    OPTIONS = "options"  # Choose one out of k available options
    BOOL = "bool"


class Parameter(dict):
    """A single configurable parameter surfaced in the study-creation UI.

    Args:
        param_name: Parameter name (also the keyword passed to the component constructor).
        param_type: One of the `ParameterType` values.
        param_default_value: Default value shown in the UI.
        help: Inline help text shown next to the field.
        help_key: Key into the translations ``.json`` file, so that the help text is
            translatable (takes precedence over ``help`` when translations are available).
    """
    def __init__(self, param_name: str, param_type: str, param_default_value: Any, help: str = None, help_key: str = None, **kwargs):
        self.name = param_name
        self.type = param_type
        self.default = param_default_value
        self.help = help
        self.help_key = help_key
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)


class AlgorithmBase(ABC):
    """Base class for recommendation algorithms.

    Subclasses are discovered automatically and offered in the fastcompare study-creation
    UI. Implementations must accept ``**kwargs`` in ``__init__``.
    """

    _my_id = "821e13b63f5a4df4ca54c30b6b0cf48f"  # MD5 hash of "AlgorithmBase"

    # Initialize method that is called after preference elicitation may be useful for some algorithms
    # TODO think if this is useful for anything except for RLprop
    # @abstractmethod
    # def initialize(elicitation_selected, elicitation_shown):
    #     pass

    @abstractmethod
    def fit():
        """Perform the initial training of the algorithm on the dataset.

        The data is supplied when the algorithm is constructed rather than passed to
        ``fit``, because some models have structure that depends on the underlying data
        (e.g. string lookups in TensorFlow). It therefore makes sense to expect that
        fitting is done on the same data the model was constructed with.
        """
        pass

    @abstractmethod
    def predict(selected_items, filter_out_items, k):
        """Recommend for a new, previously unseen user.

        The user's history is simulated from ``selected_items``. Returns a list of item
        indices for the ``k`` recommended items. None of the ``filter_out_items`` may
        appear in the result.
        """
        pass

    @classmethod
    @abstractmethod
    def name():
        """Return the algorithm's display name. Names must be unique."""
        pass

    @classmethod
    @abstractmethod
    def parameters():
        """Return the list of `Parameter` objects for this algorithm.

        These are set by the researcher when creating the user study and are passed to
        the algorithm's constructor as keyword arguments.
        """
        pass

    ### Serialization methods ###

    def load(self, instance_cache_path: str, class_cache_path: str, semi_local_cache_path: str):
        """Load internal state (default implementation uses pickle).

        More complex models may need to override this (e.g. TensorFlow models).

        Args:
            instance_cache_path: Cache for data specific to this instance (i.e. depends on
                its parameters).
            class_cache_path: A single cache shared across all parameter combinations
                (useful for static data).
            semi_local_cache_path: A per-dataset cache for the whole class (other config
                parameters are ignored).

        When in doubt, just use ``instance_cache_path`` and ignore the others.
        """
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        self.__dict__.update(attribs)
        return self

    def save(self, instance_cache_path: str, class_cache_path: str, semi_local_cache_path: str):
        """Save internal state (default implementation uses pickle).

        More complex models may need to override this. See `load` for the meaning of
        the cache-path arguments.
        """
        with open(instance_cache_path, "wb") as f:
            # Filter out "private" (starts with _) members
            pickle.dump({a: b for a, b in self.__dict__.items()
                         if (not a.startswith("_thread.")) and\
                            (not type(a).__name__.startswith("_thread.")) and\
                            (not type(b).__name__.startswith("_thread."))}, f)


class PreferenceElicitationBase(ABC):
    """Base class for preference-elicitation methods.

    Implementations must accept ``**kwargs`` in ``__init__``.
    """
    _my_id = "e1d213e443d64bf0d6bfe9a7f6f26290"  # MD5 hash of "PreferenceElicitationBase"

    @abstractmethod
    def fit():
        """Perform any dataset-dependent initialization.

        Most preference-elicitation methods do not really need this.
        """
        pass

    @abstractmethod
    def get_initial_data(movie_indices_to_ignore=[]):
        """Return the initial set of items shown to the user to select from."""
        pass

    @classmethod
    @abstractmethod
    def name():
        """Return the method's display name.

        Names must be unique; the name is shown to researchers when creating a user study
        from the fastcompare plugin.
        """
        pass

    @classmethod
    @abstractmethod
    def parameters():
        """Return the list of `Parameter` objects for this method.

        These are set by the researcher when creating the user study and are passed to the
        preference-elicitation constructor as keyword arguments.
        """
        pass

    ### Serialization methods ###

    def load(self, instance_cache_path: str, class_cache_path: str, semi_local_cache_path: str):
        """Load internal state (default implementation uses pickle).

        See `AlgorithmBase.load` for the meaning of the cache-path arguments.
        """
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        self.__dict__.update(attribs)
        return self

    def save(self, instance_cache_path: str, class_cache_path: str, semi_local_cache_path: str):
        """Save internal state (default implementation uses pickle).

        See `AlgorithmBase.load` for the meaning of the cache-path arguments.
        """
        with open(instance_cache_path, "wb") as f:
            # Filter out "private" (starts with _) members
            pickle.dump({a: b for a, b in self.__dict__.items()
                         if (not a.startswith("_thread.")) and\
                            (not type(a).__name__.startswith("_thread.")) and\
                            (not type(b).__name__.startswith("_thread."))}, f)


class DataLoaderBase(ABC):
    """Base class for dataset/domain loaders.

    Implementations must accept ``**kwargs`` in ``__init__``. Conventions:

    - ``ratings_df`` must contain ``user`` and ``item`` columns.
    - ``items_df`` must contain a ``title`` column.
    - An item **id** may be non-zero-based, whereas an item **index** is strictly
      zero-based. Use `get_item_index` / `get_item_id` to convert between them.
    """
    _my_id = "60169475d436925316ff7a2b03b52253"  # MD5 hash of "DataLoaderBase"

    @abstractmethod
    def load_data():
        """Load the data. Long-running work belongs here."""
        pass

    @property
    @abstractmethod
    def ratings_df():
        """Dataframe of interactions/ratings.

        Should contain ``user``, ``item``, and ``item_id`` (zero-based) columns. Note that
        interactions are treated as implicit feedback.
        """
        pass

    @property
    @abstractmethod
    def items_df():
        """Dataframe with item metadata. Should have ``item_id`` and ``title`` columns."""
        pass

    @property
    @abstractmethod
    def items_df_indexed():
        """Same as `items_df` but indexed by ``item``."""
        pass

    @property
    @abstractmethod
    def distance_matrix():
        """Pairwise item distance matrix (used e.g. by the ILD metric)."""
        pass

    @property
    @abstractmethod
    def rating_matrix():
        """User-by-item rating matrix."""
        pass

    @abstractmethod
    def get_item_id_image_url(item_id):
        """Return the image URL for the given item id.

        Either a remote URL (``http://…``, which can be slow) or a local one produced via
        Flask's ``url_for`` (if you place images under ``server/static/datasets/<x>/img/*.jpg``).
        """
        pass

    @abstractmethod
    def get_item_index_image_url(item_index):
        """Same as `get_item_id_image_url`, but for an item index instead of an id."""
        pass

    @abstractmethod
    def get_item_index(item_id):
        """Map an item id to its (zero-based) item index."""
        pass

    @abstractmethod
    def get_item_id(item_index):
        """Map an item index to its item id."""
        pass

    @abstractmethod
    def get_item_index_description(item_index):
        """Return a textual description for the item index (e.g. title, or title + genres)."""
        pass

    @abstractmethod
    def get_item_id_description(item_id):
        """Return a textual description for the given item id."""
        pass

    @abstractmethod
    def get_item_index_categories(item_index):
        """Return the list of categories for the given item index."""
        pass

    @abstractmethod
    def get_all_categories():
        """Return all categories available in the dataset."""
        pass

    @classmethod
    @abstractmethod
    def name():
        """Return the data loader's display name.

        Names must be unique; the name is shown to researchers when creating a user study
        from the fastcompare plugin.
        """
        pass

    @classmethod
    @abstractmethod
    def parameters():
        """Return the list of `Parameter` objects for this data loader.

        These are set by the researcher when creating the user study and are passed to the
        data loader's constructor as keyword arguments.

        Note: currently no data loaders take any parameters. If this changes, the
        semi-local cache path may need to include the parameter values.
        """
        pass

    ### Serialization methods ###

    def load(self, instance_cache_path: str, class_cache_path: str, semi_local_cache_path: str):
        """Load internal state (default implementation uses pickle).

        See `AlgorithmBase.load` for the meaning of the cache-path arguments.
        """
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        self.__dict__.update(attribs)
        return self

    def save(self, instance_cache_path: str, class_cache_path: str, semi_local_cache_path: str):
        """Save internal state (default implementation uses pickle).

        See `AlgorithmBase.load` for the meaning of the cache-path arguments.
        """
        with open(instance_cache_path, "wb") as f:
            # Filter out "private" (starts with _) members
            pickle.dump({a: b for a, b in self.__dict__.items()
                         if (not a.startswith("_thread.")) and\
                            (not type(a).__name__.startswith("_thread.")) and\
                            (not type(b).__name__.startswith("_thread."))}, f)


class EvaluationMetricBase(ABC):
    """Base class for evaluation metrics shown in the study Results view."""
    _my_id = "02afcb1d17b8eb52a4ac71f722badf5a"  # MD5 hash of "EvaluationMetricBase"

    @abstractmethod
    def evaluate(shown_items, selected_items):
        """Compute the metric for one observation.

        Takes ``shown_items`` (a list of item indices) and ``selected_items`` (a list of
        item indices) and returns a numeric evaluation result.
        """
        pass

    @classmethod
    @abstractmethod
    def name():
        """Return a unique display name for the metric."""
        pass
