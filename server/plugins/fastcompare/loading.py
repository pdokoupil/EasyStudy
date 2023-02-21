import os
import sys
import time
import pkgutil
import functools
import inspect
import plugins

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, PreferenceElicitationBase, DataLoaderBase

def is_abstract(cls):
    return bool(getattr(cls, "__abstractmethods__", False))

# We cannot test base classes directly because class types differ based on import path (e.g. Y when imported as 'from X import Y' is different than when imported as 'from A1.A2.A3.X import Y')
def is_algorithm(cls):
    return hasattr(cls, "_my_id") and cls._my_id == AlgorithmBase._my_id and not is_abstract(cls)

def is_preference_elicitation(cls):
    return hasattr(cls, "_my_id") and cls._my_id == PreferenceElicitationBase._my_id and not is_abstract(cls)

def is_data_loader(cls):
    return hasattr(cls, "_my_id") and cls._my_id == DataLoaderBase._my_id and not is_abstract(cls)

@functools.lru_cache(maxsize=None)
def load_algorithms():
    algorithms = {}
    for mod in pkgutil.walk_packages(plugins.__path__, prefix=plugins.__name__ + ".", onerror=lambda x: print("##########")):
        # if mod.name in sys.modules:
            # continue # Avoid cyclic dependencies
        imported_module = __import__(mod.name, fromlist="dummy")
        members = inspect.getmembers(imported_module) if "fastcompare.algo" in mod.name else inspect.getmembers(imported_module, inspect.isclass)
        for name, cls in members:
            if is_algorithm(cls) and cls.name() not in algorithms: # We need unique names of algorithms!
                algorithms[cls.name()] = cls

    return algorithms

@functools.lru_cache(maxsize=None)
def load_preference_elicitations():
    elicitations = {}
    for mod in pkgutil.walk_packages(plugins.__path__, prefix=plugins.__name__ + ".", onerror=lambda x: print("##########")):
        imported_module = __import__(mod.name, fromlist="dummy")
        members = inspect.getmembers(imported_module, inspect.isclass)
        for _, cls in members:
            if is_preference_elicitation(cls) and cls.name() not in elicitations: # We need unique names of elicitations!
                elicitations[cls.name()] = cls
    return elicitations

@functools.lru_cache(maxsize=None)
def load_data_loaders():
    data_loaders = {}
    for mod in pkgutil.walk_packages(plugins.__path__, prefix=plugins.__name__ + ".", onerror=lambda x: print("##########")):
        imported_module = __import__(mod.name, fromlist="dummy")
        members = inspect.getmembers(imported_module, inspect.isclass)
        for _, cls in members:
            if is_data_loader(cls) and cls.name() not in data_loaders: # We need unique names of data loaders!
                data_loaders[cls.name()] = cls
    return data_loaders