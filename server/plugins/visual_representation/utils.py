import json
import os
import time
from flask import url_for
import numpy as np
import functools

import sys
[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from common import get_abs_project_root_path

from pathlib import Path

N_EXAMPLES_PER_DATASET = 5 # We have 5 examples per dataset
PER_METHOD_ATTENTION_CHECK = True

##### Util types ######
class Example:
    def __init__(self, path, name, example_class):
        path = path.replace(os.sep, "/")
        self.path = str(path)
        self.name = str(name)
        self.example_class = example_class
    
    def toJSON(self):
        return json.loads(
            #json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)))
            json.dumps({
                #"path": self.path,
                "name": self.name,
                "example_class_name": self.example_class.name,
                "rel_path": url_for('static', filename=f'datasets/vizualizations/' + self.path.split(f'vizualizations/')[1])
            })
        )

class ExampleClass:
    def __init__(self, name, path, dataset_name):
        path = path.replace(os.sep, "/")
        self.name = str(name)
        self.path = str(path)
        #print(self.name)
        self.class_image_path = [x for x in os.listdir(Path(path).parent) if os.path.isfile(os.path.join(Path(path).parent, x)) and "." in x and x.startswith(name)]
        assert len(self.class_image_path) == 1, f"class={name}, res={self.class_image_path}"
        self.class_image_path = os.path.join(Path(path).parent.absolute(), self.class_image_path[0]).replace(os.sep, "/")
        #print(self.class_image_path)
        self.examples = []
        self.dataset_name = str(dataset_name)

        for example_name in os.listdir(path):
            example_path = os.path.join(path, example_name)
            self.examples.append(Example(example_path, os.path.splitext(example_name)[0], self))
    
   
    def toJSON(self):
        return json.loads(
            #json.dumps(self, default=lambda o: getattr(o, '__dict__', str(o)))
            json.dumps({
                "name": self.name,
                #"path": self.path,
                #"class_image_path": self.class_image_path,
                "rel_path": url_for('static', filename=f'datasets/vizualizations/' + self.path.split(f'vizualizations/')[1]),
                "class_image_rel_path": url_for('static', filename=f'datasets/vizualizations/' + self.class_image_path.split(f'vizualizations/')[1])
            })
        )

def dumper(obj):
    if hasattr(obj, "toJSON"):
        return obj.toJSON()
    else:
        return obj.__dict__

class SingleIteration:
    def __init__(self, it, example, shown_classes, method, dataset):
        self.it = int(it)
        self.example = example
        self.shown_classes = shown_classes
        self.method = str(method)
        self.dataset = str(dataset)

##### Util functions #####


@functools.cache
def get_per_dataset_classes():
    res = dict()
    for dataset in get_datasets():
        res[dataset] = dict()
        dataset_dir = os.path.join(get_dataset_base_dir(), "vizualizations", dataset)
        for method in list_dirs(dataset_dir):
            res[dataset][method] = dict()
            for class_name in list_dirs(os.path.join(get_dataset_base_dir(), "vizualizations", dataset, method)):
                class_path = os.path.join(get_dataset_base_dir(), "vizualizations", dataset, method, class_name)    
                res[dataset][method][class_name] = ExampleClass(class_name, class_path, dataset)
                #print(f"Listing: ", class_path)
                #res[dataset][cls] = list(map(lambda x: x.split(".")[0], os.listdir(class_path)))
    return res

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
            methods = list(set(methods).union(found_methods))
    assert methods is not None
    return methods

# Return all available datasets
@functools.cache
def get_datasets():
    datasets_base_dir = get_dataset_base_dir()
    return list_dirs(os.path.join(datasets_base_dir, "vizualizations"))



@functools.cache
def get_per_dataset_union_examples():
    x = get_per_dataset_classes()
    res = dict()
    for dataset, methods in x.items():
        res[dataset] = []
        for method, classes in methods.items():
            res[dataset] = list({example_name: (example_class, example_name) for example_class, example_name in sum(map(lambda x: [(y.example_class.name, Path(y.path).stem) for y in x.examples], classes.values()), res[dataset])}.values())

    return res

# Builds list of steps for the user ("permutaton")
def build_permutation():
    # Get all available methods (based on avilable folders)
    methods = get_methods()
    # Get all available datasets (based on avilable folders)
    datasets = get_datasets()

    # Shuffle methods and datasets
    np.random.shuffle(methods)
    np.random.shuffle(datasets)

    # For each dataset, get all available examples (union over files in subfolders in each dataset folder)
    per_dataset_available_examples = get_per_dataset_union_examples()
    per_dataset_selected_example_ids = dict()
    per_dataset_selected_example_shown_class_names = dict()

    # Get available classes for each dataset and each method
    per_dataset_classes = get_per_dataset_classes()

    # Iterate over datasets and get examples to be shown for each of them
    # For a single dataset, same examples are shown by all methods
    # We do the same for the shown classes, except that the order of classes may change between iterations
    for dataset in datasets:
        # Sample 5 examples for each dataset without replacement
        indices = np.random.choice(range(len(per_dataset_available_examples[dataset])), size=N_EXAMPLES_PER_DATASET, replace=False)
        assert len(indices) == N_EXAMPLES_PER_DATASET
        per_dataset_selected_example_ids[dataset] = {}
        per_dataset_selected_example_shown_class_names[dataset] = {}

        # Convert to dict of lists of IDs
        for x, y in [per_dataset_available_examples[dataset][idx] for idx in indices]:
            per_dataset_selected_example_ids[dataset].setdefault(x, []).append(y)
            # For every tuple x, y (class, example id/name) we generate classes that will be shown for users to select from
            
            # For a given dataset, get available classes (take first method and lookup classes in its folder)
            classes = list(per_dataset_classes[dataset][list(per_dataset_classes[dataset].keys())[0]].values())
            # If we have less than or equal to 4 classes, take all of them (convert classes to names)
            # Otherwise take class corresponding to selected example and sample remaining three classes on random
            shown_class_names = [c.name for c in classes] if len(classes) <= 4 else [x] + np.random.choice([c.name for c in classes if c.name != x], size=3, replace=False).tolist()
            per_dataset_selected_example_shown_class_names[dataset][(x, y)] = shown_class_names

    permutation = []

    it = 0

    # Iterate over methods and start building the sequence of steps (for each method, each dataset, each example)
    for method in methods:
        # If attention check is enabled (it is, by default)
        if PER_METHOD_ATTENTION_CHECK:
            # Randomly select dataset for which we will make the attention check
            # receipesWeighted is valid only for SOM
            valid_datasets = [d for d, v in per_dataset_classes.items() if method in v]
            # Chose a single dataset
            attention_check_dataset = np.random.choice(valid_datasets)
            
        # Iterate over datasets
        for dataset in datasets:
            # If the method does not contain anything for the given dataset (e.g. recipesWeighted for other method than SOM), we skip
            if not method in per_dataset_classes[dataset]:
                continue
            
            # Get classes that are available for the given dataset and method
            classes = list(per_dataset_classes[dataset][method].values())
            class_name_to_class = {c.name : c for c in classes}
            assert type(classes[0]) == ExampleClass, type(classes[0])
            
            # Terribly inefficient iteration, we iterate classes and check if we have chosen any example with that class
            # We have the example IDs but we need to get example instances itself, this iteration just finds for each ID the corresponding object instance
            examples = []
            # For each examples we have a certain number of classes that will be shown to the user
            example_classes = []
            for cls in classes:
                # Check if there is selected example having given class cls
                if cls.name in per_dataset_selected_example_ids[dataset]:
                    # We know we have such an example, but we need to get reference to it
                    for ex in cls.examples:
                        if ex.name in per_dataset_selected_example_ids[dataset][cls.name]:
                            # We found the example we searched for
                            examples.append(ex)
                            # Do reverse conversion, from class name to class instance
                            example_classes.append([class_name_to_class[c_name] for c_name in per_dataset_selected_example_shown_class_names[dataset][(cls.name, ex.name)]])
                            

            # Verify we have exactly N_EXAMPLES_PER_DATASET for this method and dataset
            assert len(examples) == N_EXAMPLES_PER_DATASET

            # Iterate over examples and push them to the sequence
            for example, example_cls in zip(examples, example_classes):
                assert type(example.example_class) == ExampleClass
                # Decide which classes to show for the example
                # If we have <= 4 classes available, we show all of them, otherwise we show correct class + 3 randomly chosen classes
                #shown_classes = classes if len(classes) <= 4 else [example.example_class] + np.random.choice([c for c in classes if c != example.example_class], size=3, replace=False).tolist()
                #shown_classes = shown_classes[:]
                
                shown_classes = example_cls[:]
                # Shuffle classes so their order is randomized
                np.random.shuffle(shown_classes)

                permutation.append(SingleIteration(it, example, shown_classes, method, dataset))
                it += 1

            # If attention check is enabled and we are currently processing the dataset for which the attention check should be added, we add it
            # The check is done in this way because we want to position the attention check at the very end of PER_METHOD_ATTENTION_CHECK steps for each method
            if PER_METHOD_ATTENTION_CHECK and dataset == attention_check_dataset:
                ##print("### Injecting attention check")
                # Select single class for attention check
                attention_check_class = np.random.choice(classes)
                # Create artificial example that points to the class representation image
                artificial_example = Example(attention_check_class.class_image_path, f"{method}_{dataset}_{attention_check_class.name}_atn", attention_check_class)

                # Chose classes to shown below 
                atn_check_shown_classes = classes if len(classes) <= 4 else [artificial_example.example_class] + np.random.choice([c for c in classes if c.name != artificial_example.example_class.name], size=3, replace=False).tolist()
                atn_check_shown_classes = atn_check_shown_classes[:]
                
                # Shuffle classes for the attention check
                np.random.shuffle(atn_check_shown_classes)

                permutation.append(SingleIteration(it, artificial_example, atn_check_shown_classes, method, dataset))
                it += 1
            
    return permutation


if __name__ == "__main__":
    print(get_datasets())
    print("")
    print(get_methods())
    print("")
    print(get_per_dataset_classes())
    print("")
    print(get_per_dataset_union_examples())
    print("")
    print("PERMUTATION")
    print("")
    start_time = time.perf_counter()
    perm = build_permutation()
    took = time.perf_counter() - start_time
    print(f"Perm length = {len(perm)}")

    for x in perm:
        print(f"Iteration = {x.it}")
        print(f"\tMethod = {x.method}")
        print(f"\tDataset = {x.dataset}")
        print(f"\tExample = {x.example.path}")
        print(f"\tShown classes = {[y.name for y in x.shown_classes]}")
        print(f"\tShown classes path = {[y.class_image_path for y in x.shown_classes]}")
    
    print(f"Generation took: {took}")