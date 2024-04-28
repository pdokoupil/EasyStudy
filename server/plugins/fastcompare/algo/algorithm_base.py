from abc import ABC, abstractmethod
import pickle


algorithm_registry = []

def register_algorithm(algo_class):
    assert issubclass(algo_class, AlgorithmBase)
    if not algo_class in algorithm_registry:
        algorithm_registry.append(algo_class)
    print(f"Successfully registered")

# Class for hyperparamters/configurable parameters of the algorithms
class ParameterType:
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    OPTIONS = "options" # Choose one out of k available options
    BOOL = "bool"

class Parameter(dict):
    def __init__(self, param_name, param_type, param_default_value, help=None, help_key=None, **kwargs):
        self.name = param_name
        self.type = param_type
        self.default = param_default_value
        self.help = help
        self.help_key = help_key # help_key refers to .json with translations, so that kind of help is translatable
        self.__dict__.update(kwargs)
        dict.__init__(self, self.__dict__)

# Base classes must take **kwargs in __init__
class AlgorithmBase(ABC):
    
    _my_id = "821e13b63f5a4df4ca54c30b6b0cf48f" # MD5 hash of "AlgorithmBase"

    # Initialize method that is called after preference elicitation may be useful for some algorithms
    # TODO think if this is useful for anything except for RLprop
    # @abstractmethod
    # def initialize(elicitation_selected, elicitation_shown):
    #     pass

    @abstractmethod
    # Data should be specified when algorithm is constructed, not passed to fit
    # the reason is that some models have structure dependent on the underlying data (e.g. string lookups in tensorflow etc.)
    # so it makes sense to expect fitting is done at same data as construction
    # This method should perform initial training on the dataset
    def fit():
        pass

    # Performs prediction for new, previously unknown user, by simulating his history using a list of selected items.
    # It should return list of item indices of the k recommended items. Note, that none of the filter_out_items can be present in the result
    @abstractmethod
    def predict(selected_items, filter_out_items, k):
        pass

    # Names have to be unique!
    @classmethod
    @abstractmethod
    def name():
        pass

    # Return list of parameters (see ParameterType type) that will be set by the user when creating the user study and passed to the
    # Algorithm's constructor
    @classmethod
    @abstractmethod
    def parameters():
        pass

    ### Serialization methods ###

    # Load internal state, default implementation using pickle
    # more complex models may need to override this behavior (e.g. tensorflow models)
    # Instance_cache_path is for data specific for each instance (e.g. depends on parameters)
    # while class_cache_path is single cache for all combinations (useful for static data)
    # When in doubt, just use instance_cache_path and ignore class_cache_path
    def load(self, instance_cache_path, class_cache_path):
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        self.__dict__.update(attribs) 

    # Save internal state, default implementation using pickle
    # more complex models may need to override this behavior
    # Instance_cache_path is for data specific for each instance (e.g. depends on parameters)
    # while class_cache_path is single cache for all combinations (useful for static data)
    # When in doubt, just use instance_cache_path and ignore class_cache_path
    def save(self, instance_cache_path, class_cache_path):
        with open(instance_cache_path, "wb") as f:
            pickle.dump(self.__dict__, f)

# Base classes must take **kwargs in __init__
class PreferenceElicitationBase(ABC):
    _my_id = "e1d213e443d64bf0d6bfe9a7f6f26290" # MD5 hash of "PreferenceElicitationBase"

    # Perform any sort of dataset-dependent initialization (most of the preference elicitation methods does not really need this)
    @abstractmethod
    def fit():
        pass

    # Returns the initial data shown to the user that the user is asked to select from
    @abstractmethod
    def get_initial_data(movie_indices_to_ignore=[]):
        pass

    # Names have to be unique! Will be displayed to the users when creating user study from fastcompare plugin
    @classmethod
    @abstractmethod
    def name():
        pass
    
    # Return list of parameters (see ParameterType type) that will be set by the user when creating the user study and passed to the
    # Preference elicitation's constructor
    @classmethod
    @abstractmethod
    def parameters():
        pass

    ### Serialization methods ###

    # Load internal state, default implementation using pickle
    # more complex models may need to override this behavior (e.g. tensorflow models)
    # Instance_cache_path is for data specific for each instance (e.g. depends on parameters)
    # while class_cache_path is single cache for all combinations (useful for static data)
    # When in doubt, just use instance_cache_path and ignore class_cache_path
    def load(self, instance_cache_path, class_cache_path):
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        self.__dict__.update(attribs) 

    # Save internal state, default implementation using pickle
    # more complex models may need to override this behavior
    # Instance_cache_path is for data specific for each instance (e.g. depends on parameters)
    # while class_cache_path is single cache for all combinations (useful for static data)
    # When in doubt, just use instance_cache_path and ignore class_cache_path
    def save(self, instance_cache_path, class_cache_path):
        with open(instance_cache_path, "wb") as f:
            pickle.dump(self.__dict__, f)

# Base classes must take **kwargs in __init__
# there should be user, item columns in the ratings_df
# there should also be title column in the items_df
# Item id is possibily non-zero based id, while item_index is strictly zero based
class DataLoaderBase(ABC):
    _my_id = "60169475d436925316ff7a2b03b52253" # MD5 hash of "DataLoaderBase"

    # Load the data, here you can perform long running stuff
    @abstractmethod
    def load_data():
        pass

    # Returns dataframe with the interactions/ratings (be aware that implicit feedback is now considered)
    # There should be "user", "item", and "item_id" (zero-based) columns in the dataframe
    @property
    @abstractmethod
    def ratings_df():
        pass

    # Returns dataframe with information about items. Should have item_id, and title columns
    @property
    @abstractmethod
    def items_df():
        pass

    # Same as items_df but using item as an index
    @property
    @abstractmethod
    def items_df_indexed():
        pass

    # Return image url for the given item id
    # Either remote URL (http://) (could be slow)
    # Or local, already processed via flask's url_for (if you place images into server/static/datasets/x/img/*.jpg)
    @abstractmethod
    def get_item_id_image_url(item_id):
        pass

    # Same as above, but for given item index instead of item id
    @abstractmethod    
    def get_item_index_image_url(item_index):
        pass

    # Map item id to item index
    @abstractmethod
    def get_item_index(item_id):
        pass

    # Map item index to item id
    @abstractmethod
    def get_item_id(item_index):
        pass

    # Return textual description for the given item index (e.g. title or title concatenated with genres, etc.)
    @abstractmethod
    def get_item_index_description(item_index):
        pass

    # Return textual description for the given item id
    @abstractmethod
    def get_item_id_description(item_id):
        pass
    
    # For a given item index, return list of its categories
    @abstractmethod
    def get_item_index_categories(item_index):
        pass

    # Return all available categories in the dataset
    @abstractmethod
    def get_all_categories():
        pass

    # Names have to be unique! Return data loader name that will be displayed to the user when creating user study using fastcompare plugin
    @classmethod
    @abstractmethod
    def name():
        pass

    # Return list of parameters (see ParameterType type) that will be set by the user when creating the user study and passed to the
    # Data loader's constructor
    @classmethod
    @abstractmethod
    def parameters():
        pass

    ### Serialization methods ###

    # Load internal state, default implementation using pickle
    # more complex models may need to override this behavior (e.g. tensorflow models)
    # Instance_cache_path is for data specific for each instance (e.g. depends on parameters)
    # while class_cache_path is single cache for all combinations (useful for static data)
    # When in doubt, just use instance_cache_path and ignore class_cache_path
    def load(self, instance_cache_path, class_cache_path):
        with open(instance_cache_path, "rb") as f:
            attribs = pickle.load(f)
        self.__dict__.update(attribs) 

    # Save internal state, default implementation using pickle
    # more complex models may need to override this behavior
    # Instance_cache_path is for data specific for each instance (e.g. depends on parameters)
    # while class_cache_path is single cache for all combinations (useful for static data)
    # When in doubt, just use instance_cache_path and ignore class_cache_path
    def save(self, instance_cache_path, class_cache_path):
        with open(instance_cache_path, "wb") as f:
            pickle.dump(self.__dict__, f)

class Algo1:
    pass

class Algo2(AlgorithmBase):

    def fit():
        pass

    def predict(self, selected_items, filter_out_items, k):
        pass

class Algo3(Algo1):
    pass

def get_functions_and_methods(path):
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

