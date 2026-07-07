import os
import sys
import time
import pkgutil
import functools
import inspect
import plugins

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase, PreferenceElicitationBase, DataLoaderBase, EvaluationMetricBase

def is_abstract(cls):
    return bool(getattr(cls, "__abstractmethods__", False))

# We cannot test base classes directly because class types differ based on import path (e.g. Y when imported as 'from X import Y' is different than when imported as 'from A1.A2.A3.X import Y')
def is_algorithm(cls):
    return hasattr(cls, "_my_id") and cls._my_id == AlgorithmBase._my_id and not is_abstract(cls)

def is_preference_elicitation(cls):
    return hasattr(cls, "_my_id") and cls._my_id == PreferenceElicitationBase._my_id and not is_abstract(cls)

def is_data_loader(cls):
    return hasattr(cls, "_my_id") and cls._my_id == DataLoaderBase._my_id and not is_abstract(cls)

def is_evaluation_metric(cls):
    return hasattr(cls, "_my_id") and cls._my_id == EvaluationMetricBase._my_id and not is_abstract(cls)


def _iter_plugin_modules():
    """Yield every importable module under `plugins`, skipping ones whose (optional)
    dependencies are not installed.

    Algorithm/loader/metric discovery walk-imports the whole plugins tree. Some modules
    (e.g. the TF-Recommenders / VAE algorithms) import heavy optional extras at module
    top level. On a lightweight install those imports raise ImportError; we skip such
    modules so their components simply do not appear, instead of crashing all discovery.
    Any other import-time error is reported but likewise skipped so one broken plugin
    cannot take down the study-creation UI.
    """
    for mod in pkgutil.walk_packages(plugins.__path__, prefix=plugins.__name__ + ".", onerror=lambda x: print(f"walk error in {x}")):
        try:
            yield mod, __import__(mod.name, fromlist="dummy")
        except ImportError as exc:
            print(f"[loading] skipping '{mod.name}' — optional dependency missing: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not break discovery
            print(f"[loading] skipping '{mod.name}' — import failed: {exc!r}")


@functools.lru_cache(maxsize=None)
def load_algorithms():
    algorithms = {}
    for mod, imported_module in _iter_plugin_modules():
        members = inspect.getmembers(imported_module) if "fastcompare.algo" in mod.name else inspect.getmembers(imported_module, inspect.isclass)
        for name, cls in members:
            if is_algorithm(cls) and cls.name() not in algorithms: # We need unique names of algorithms!
                algorithms[cls.name()] = cls

    return algorithms

@functools.lru_cache(maxsize=None)
def load_preference_elicitations():
    elicitations = {}
    for _mod, imported_module in _iter_plugin_modules():
        members = inspect.getmembers(imported_module, inspect.isclass)
        for _, cls in members:
            if is_preference_elicitation(cls) and cls.name() not in elicitations: # We need unique names of elicitations!
                elicitations[cls.name()] = cls
    return elicitations

@functools.lru_cache(maxsize=None)
def load_data_loaders():
    data_loaders = {}
    for _mod, imported_module in _iter_plugin_modules():
        members = inspect.getmembers(imported_module, inspect.isclass)
        for _, cls in members:
            if is_data_loader(cls) and cls.name() not in data_loaders: # We need unique names of data loaders!
                data_loaders[cls.name()] = cls
    return data_loaders


@functools.lru_cache(maxsize=None)
def load_evaluation_metrics():
    evaluation_metrics = {}
    for _mod, imported_module in _iter_plugin_modules():
        members = inspect.getmembers(imported_module, inspect.isclass)
        for _, cls in members:
            if is_evaluation_metric(cls) and cls.name() not in evaluation_metrics: # We need unique names of evaluation metrics!
                evaluation_metrics[cls.name()] = cls
    return evaluation_metrics