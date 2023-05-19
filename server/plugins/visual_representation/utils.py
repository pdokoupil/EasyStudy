import os
import numpy as np
import functools

import sys
[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from common import get_abs_project_root_path


N_EXAMPLES_PER_METHOD = 5 # We have 5 examples per method
PER_METHOD_ATTENTION_CHECK = True

##### Util types ######
class Example:
    def __init__(self):
        self.path = path
        self.name = name
        self.cls = cls

class ExampleClass:
    pass


##### Util functions #####

# Return all available methods

def list_dirs(path):
    return [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

def get_dataset_base_dir():
    return os.path.join(get_abs_project_root_path(), 'static', 'datasets')

@functools.cache
def get_methods():
    datasets_base_dir = get_dataset_base_dir()
    methods = None
    for x in list_dirs(os.path.join(datasets_base_dir, "vizualizations")):
        found_methods = list_dirs(os.path.join(datasets_base_dir, "vizualizations", x))
        if methods is None:
            methods = found_methods
        else:
            assert set(methods) == set(found_methods), f"Same methods for all datasets: {methods}, {x}, {found_methods}"
    assert methods is not None
    return methods

# Return all available datasets
@functools.cache
def get_datasets():
    datasets_base_dir = get_dataset_base_dir()
    return list_dirs(os.path.join(datasets_base_dir, "vizualizations"))

@functools.cache
def get_per_dataset_classes():
    res = dict()
    for dataset in get_datasets():
        res[dataset] = dict()
        dataset_dir = os.path.join(get_dataset_base_dir(), "vizualizations", dataset)
        for method in list_dirs(dataset_dir):
            for cls in list_dirs(os.path.join(get_dataset_base_dir(), "vizualizations", dataset, method)):
                class_path = os.path.join(get_dataset_base_dir(), "vizualizations", dataset, method, cls)
                print(f"Listing: ", class_path)
                res[dataset][cls] = list(map(lambda x: x.split(".")[0], os.listdir(class_path)))
    return res

@functools.cache
def get_per_dataset_union_examples():
    x = get_per_dataset_classes()
    res = dict()
    for dataset, classes in x.items():
        res[dataset] = set(sum(classes.values(), []))
    return res

# Builds permutation of steps for the user
def build_permutation():
    methods = get_methods()
    datasets = get_datasets()

    # Shuffle
    np.random.shuffle(methods)
    np.random.shuffle(datasets)

    per_dataset_available_examples = get_per_dataset_union_examples()
    per_dataset_selected_examples = dict()

    for dataset in datasets:
        per_dataset_selected_examples[dataset] = np.random.choice(per_dataset_available_examples[dataset], replace=False)

    for method in methods:
        for dataset in datasets:
            examples = per_dataset_selected_examples[dataset]
            if PER_METHOD_ATTENTION_CHECK:
                examples.append()


if __name__ == "__main__":
    print(get_datasets())
    print("")
    print(get_methods())
    print("")
    print(get_per_dataset_classes())
    print("")
    print(get_per_dataset_union_examples())
    print("")