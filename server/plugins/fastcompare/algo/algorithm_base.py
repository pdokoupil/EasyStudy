from __future__ import annotations

from abc import ABC, abstractmethod
import pickle
import pandas as pd
import numpy as np
import numpy.typing as npt


algorithm_registry = []

def register_algorithm(algo_class: type[AlgorithmBase]):
    assert issubclass(algo_class, AlgorithmBase)
    if algo_class not in algorithm_registry:
        algorithm_registry.append(algo_class)
    print("Successfully registered")

class ParameterType:
    """Hyperparameter/configurable parameter types for algorithms."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    OPTIONS = "options"
    """Choose one out of k available options. TODO how to define the options?"""
    BOOL = "bool"

class Parameter(dict):
    """A configurable parameter for an algorithm or data loader."""

    def __init__(self, param_name: str, param_type: str, param_default_value: object, help: str | None = None, help_key: str | None = None, **kwargs):
        self.name = param_name
        self.type = param_type
        self.default = param_default_value
        self.help = help
        self.help_key = help_key # help_key refers to .json with translations, so that kind of help is translatable
        vars(self).update(kwargs)
        dict.__init__(self, vars(self))

class AlgorithmBase(ABC):
    """Base class for recommendation algorithms.

    Subclasses must take **kwargs in __init__.
    """

    _my_id = "821e13b63f5a4df4ca54c30b6b0cf48f"
    """MD5 hash of "AlgorithmBase"."""

    @abstractmethod
    def __init__(self, loader: DataLoaderBase, **kwargs):
        pass

    # Initialize method that is called after preference elicitation may be useful for some algorithms
    # TODO think if this is useful for anything except for RLprop
    # @abstractmethod
    # def initialize(elicitation_selected, elicitation_shown):
    #     pass

    @abstractmethod
    def fit(self):
        """Perform initial training on the dataset.

        Data should be specified when the algorithm is constructed, not passed
        to fit, because some models have structure dependent on the underlying
        data (e.g. string lookups in TensorFlow), so it makes sense to expect
        fitting is done on the same data as construction.
        """
        pass

    @abstractmethod
    def predict(self, selected_items: list[int], filter_out_items: list[int], k: int) -> list[int]:
        """Predict recommendations for a new, previously unknown user.

        Simulates the user's history using a list of selected items and returns
        the item indices of the k recommended items. None of the filter_out_items
        can be present in the result.

        :param selected_items: list of item that the user has selected during preference elicitation, match with ``ratings_df.item``
        :param filter_out_items: list of items that should not be recommended, match with ``ratings_df.item``
        :param k: number of items to recommend
        :return: list of items to recommend
        """
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the unique name of the algorithm. Names have to be unique!"""
        pass

    @classmethod
    @abstractmethod
    def parameters(cls) -> list[Parameter]:
        """Return the list of parameters (see ``Parameter``) set by the administrator
        when creating the user study and passed to the algorithm's constructor."""
        pass

    ### Serialization methods ###

    def load(self, instance_cache_path: str, class_cache_path: str):
        """Load internal state.

        Default implementation uses pickle; more complex models may need to
        override this behavior (e.g. TensorFlow models).

        ``instance_cache_path`` is for data specific to each instance (e.g.
        depends on parameters), while ``class_cache_path`` is a single cache for
        all combinations (useful for static data). When in doubt, just use
        ``instance_cache_path`` and ignore ``class_cache_path``.
        """
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        vars(self).update(attribs)

    def save(self, instance_cache_path: str, class_cache_path: str):
        """Save internal state.

        Default implementation uses pickle; more complex models may need to
        override this behavior.

        ``instance_cache_path`` is for data specific to each instance (e.g.
        depends on parameters), while ``class_cache_path`` is a single cache for
        all combinations (useful for static data). When in doubt, just use
        ``instance_cache_path`` and ignore ``class_cache_path``.
        """
        with open(instance_cache_path, "wb") as f:
            pickle.dump(vars(self), f)

class PreferenceElicitationBase(ABC):
    """Base class for preference elicitation methods.

    Subclasses must take ``**kwargs`` in ``__init__``.
    """

    _my_id = "e1d213e443d64bf0d6bfe9a7f6f26290"
    """MD5 hash of "PreferenceElicitationBase"."""

    @abstractmethod
    def fit(self):
        """Perform any sort of dataset-dependent initialization.

        Most preference elicitation methods do not really need this.
        """
        pass

    @abstractmethod
    def get_initial_data(self, movie_indices_to_ignore: list[int] | None = None) -> npt.NDArray[np.integer]:
        """Return the initial data shown to the user that the user is asked to select from."""
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the unique name. Names have to be unique!

        Will be displayed to the users when creating a user study from the
        fastcompare plugin.
        """
        pass

    @classmethod
    @abstractmethod
    def parameters(cls) -> list[Parameter]:
        """Return the list of parameters (see ``Parameter``) set by the user when
        creating the user study and passed to the preference elicitation's constructor."""
        pass

    ### Serialization methods ###

    def load(self, instance_cache_path: str, class_cache_path: str):
        """Load internal state.

        Default implementation uses pickle; more complex models may need to
        override this behavior (e.g. TensorFlow models).

        ``instance_cache_path`` is for data specific to each instance (e.g.
        depends on parameters), while ``class_cache_path`` is a single cache for
        all combinations (useful for static data). When in doubt, just use
        ``instance_cache_path`` and ignore ``class_cache_path``.
        """
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        vars(self).update(attribs)

    def save(self, instance_cache_path: str, class_cache_path: str):
        """Save internal state.

        Default implementation uses pickle; more complex models may need to
        override this behavior.

        ``instance_cache_path`` is for data specific to each instance (e.g.
        depends on parameters), while ``class_cache_path`` is a single cache for
        all combinations (useful for static data). When in doubt, just use
        ``instance_cache_path`` and ignore ``class_cache_path``.
        """
        with open(instance_cache_path, "wb") as f:
            pickle.dump(vars(self), f)

class DataLoaderBase(ABC):
    """Base class for dataset loaders.

    Subclasses must take ``**kwargs`` in ``__init__``. There should be ``user`` and
    ``item`` columns in the ratings_df, and a ``title`` column in the items_df.
    Item id is possibly a non-zero based id, while item_index is strictly zero
    based.
    """

    _my_id = "60169475d436925316ff7a2b03b52253"
    """MD5 hash of "DataLoaderBase"."""

    @abstractmethod
    def load_data(self):
        """Load the data. Long-running stuff can be performed here."""
        pass

    @property
    @abstractmethod
    def ratings_df(self) -> pd.DataFrame:
        """Return the dataframe with the interactions/ratings.

        Columns:
            - user: int64
            - item_id: int64 (zero-based)
            - rating: float64
            - timestamp: int64
            - ratings_per_year: float64
            - item: int64

        TODO specify difference between item_id and item
        """
        pass

    @property
    @abstractmethod
    def items_df(self) -> pd.DataFrame:
        """Return the dataframe with information about items.

        Should have at least ``item_id`` and ``title`` columns. MLDataLoaderWrapper adds some more.
        """
        pass

    @property
    @abstractmethod
    def items_df_indexed(self) -> pd.DataFrame:
        """Same as ``items_df``, but ``item_id`` is the index column."""
        pass

    @abstractmethod
    def get_item_id_image_url(self, item_id: int) -> str:
        """Return the image URL for the given item id.

        Either a remote URL (http://) (could be slow), or local, already
        processed via Flask's url_for (if you place images into
        server/static/datasets/x/img/*.jpg).
        """
        pass

    @abstractmethod
    def get_item_index_image_url(self, item_index: int) -> str:
        """Return the image URL for the given item index instead of item id."""
        pass

    @abstractmethod
    def get_item_index(self, item_id: int) -> int:
        """Map item id to item index."""
        pass

    @abstractmethod
    def get_item_id(self, item_index: int) -> int:
        """Map item index to item id."""
        pass

    @abstractmethod
    def get_item_index_description(self, item_index: int) -> str:
        """Return a textual description for the given item index.

        E.g. title or title concatenated with genres, etc.
        """
        pass

    @abstractmethod
    def get_item_id_description(self, item_id: int) -> str:
        """Return a textual description for the given item id."""
        pass

    @abstractmethod
    def get_item_index_categories(self, item_index: int) -> list[str]:
        """For a given item index, return list of its categories."""
        pass

    @abstractmethod
    def get_all_categories(self) -> set[str]:
        """Return all available categories in the dataset."""
        pass

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the data loader name.

        Names have to be unique! Will be displayed to the user when creating a
        user study using the fastcompare plugin.
        """
        pass

    @classmethod
    @abstractmethod
    def parameters(cls) -> list[Parameter]:
        """Return the list of parameters (see ``Parameter``) set by the user when
        creating the user study and passed to the data loader's constructor."""
        pass

    ### Serialization methods ###

    def load(self, instance_cache_path: str, class_cache_path: str):
        """Load internal state.

        Default implementation uses pickle; more complex models may need to
        override this behavior (e.g. TensorFlow models).

        ``instance_cache_path`` is for data specific to each instance (e.g.
        depends on parameters), while ``class_cache_path`` is a single cache for
        all combinations (useful for static data). When in doubt, just use
        ``instance_cache_path`` and ignore ``class_cache_path``.
        """
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        vars(self).update(attribs)

    def save(self, instance_cache_path: str, class_cache_path: str):
        """Save internal state.

        Default implementation uses pickle; more complex models may need to
        override this behavior.

        ``instance_cache_path`` is for data specific to each instance (e.g.
        depends on parameters), while ``class_cache_path`` is a single cache for
        all combinations (useful for static data). When in doubt, just use
        ``instance_cache_path`` and ignore ``class_cache_path``.
        """
        with open(instance_cache_path, "wb") as f:
            pickle.dump(vars(self), f)


def get_functions_and_methods(path: str) -> list:
    """
    Given a .py file path - returns a list with all functions and methods in it.

    Source: https://stackoverflow.com/q/73239026/256662
    """
    import ast

    with open(path) as file:
        node = ast.parse(file.read())

    def show_info(functionNode):
        function_rep = ''
        function_rep = functionNode.name + '('

        for arg in functionNode.args.args:
            function_rep += arg.arg + ','

        function_rep = function_rep.rstrip(function_rep[-1])
        function_rep += ')'
        return function_rep

    result = []
    functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
    print(classes)
    for function in functions:
        result.append(show_info(function))

    for class_ in classes:
        methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
        for method in methods:
            result.append((class_.name + '.' + show_info(method)))
    return result
