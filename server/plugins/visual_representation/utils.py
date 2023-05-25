import json
import os
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

    # def __str__(self):
    #     return self.name

    # def __repr__(self):
    #     return self.name
    
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
    
    # def __str__(self):
    #     return self.toJSON()

    # def __repr__(self):
    #     return self.toJSON()
    
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

    # def __str__(self):
    #     return self.toJSON()
    
    # def __repr__(self):
    #     return self.toJSON()
    
    
       #return json.loads(
       #     json.dumps({"it": self.it, "method": self.method, "dataset": self.dataset, "shown_classes": self.example}, default=lambda o: o.__dict__)
       # )
    
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
            # res[dataset] = list(
            #     set(
            #         sum(
            #             map(lambda x: [(Path(y.path).stem, y.example_class.name) for y in x.examples], classes.values()), res[dataset]
            #         )
            #     )
            #     unique()
            #     )

    return res

# Builds permutation of steps for the user
def build_permutation():
    methods = get_methods() #['table-t', 'radial'] #get_methods()
    datasets = get_datasets()

    # Shuffle
    np.random.shuffle(methods)
    np.random.shuffle(datasets)

    per_dataset_available_examples = get_per_dataset_union_examples()
    per_dataset_selected_example_ids = dict()
    

    for dataset in datasets:
        indices = np.random.choice(range(len(per_dataset_available_examples[dataset])), size=N_EXAMPLES_PER_DATASET, replace=False)
        assert len(indices) == N_EXAMPLES_PER_DATASET
        per_dataset_selected_example_ids[dataset] = {}
        for x, y in [per_dataset_available_examples[dataset][idx] for idx in indices]:
            per_dataset_selected_example_ids[dataset].setdefault(x, []).append(y)
        #per_dataset_selected_example_ids[dataset] = dict(np.random.choice(per_dataset_available_examples[dataset], size=N_EXAMPLES_PER_DATASET, replace=False).tolist())

    permutation = []

    it = 0

    per_dataset_classes = get_per_dataset_classes()

    for method in methods:
        if PER_METHOD_ATTENTION_CHECK:
            # Randomly select dataset for which we will make the attention check
            attention_check_dataset = np.random.choice([d for d in datasets if d != "recipesWeighted"])
            #print(F"Attention check dataset = {attention_check_dataset}")
            

        for dataset in datasets:
            if not method in per_dataset_classes[dataset]:
                continue
            #examples = per_dataset_selected_examples[dataset]
            classes = list(per_dataset_classes[dataset][method].values())
            assert type(classes[0]) == ExampleClass, type(classes[0])
            
            examples = [] #[x for x in classes if x.]
            for cls in classes:
                #print(f"Iterating class = {cls.name}")
                if cls.name in per_dataset_selected_example_ids[dataset]:
                    #print(f"Is in")
                    for ex in cls.examples:
                        if ex.name in per_dataset_selected_example_ids[dataset][cls.name]:
                            examples.append(ex)

            #print(per_dataset_selected_example_ids[dataset])
            assert len(examples) == N_EXAMPLES_PER_DATASET

            for example in examples:
                assert type(example.example_class) == ExampleClass
                shown_classes = classes if len(classes) <= 4 else [example.example_class] + np.random.choice([c for c in classes if c != example.example_class], size=3, replace=False).tolist()
                shown_classes = shown_classes[:]
                # Shuffle classes
                np.random.shuffle(shown_classes)

                permutation.append(SingleIteration(it, example, shown_classes, method, dataset))
                it += 1

            if PER_METHOD_ATTENTION_CHECK and dataset == attention_check_dataset:
                ##print("### Injecting attention check")
                # Select single class for attention check
                attention_check_class = np.random.choice(classes)
                # Create artificial example that points to the class representation image
                example = Example(attention_check_class.class_image_path, f"{method}_{dataset}_{attention_check_class.name}_atn", attention_check_class)
                permutation.append(SingleIteration(it, example, shown_classes, method, dataset))
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
    perm = build_permutation()
    print(f"Perm length = {len(perm)}")

    for x in perm:
        print(f"Iteration = {x.it}")
        print(f"\tMethod = {x.method}")
        print(f"\tDataset = {x.dataset}")
        print(f"\tExample = {x.example.path}")
        print(f"\tShown classes = {[y.name for y in x.shown_classes]}")
        print(f"\tShown classes path = {[y.class_image_path for y in x.shown_classes]}")
