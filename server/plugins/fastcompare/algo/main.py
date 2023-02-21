import inspect
import sys
[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]
from plugins.fastcompare.algo.algorithm_base import AlgorithmBase


if __name__ == "__main__":
    from pathlib import Path
    import glob
    import os


    sys.path.append("../..")

    if __name__ == "__main__":
        mod = __import__("custom_algo", "some_algo")
        print(inspect.getmembers(mod, inspect.isclass))
        print("OK2")
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            print(issubclass(cls, AlgorithmBase))

    assert False

    x = Path(__file__).resolve()
    while x.name != "plugins":
        x = x.parent
    
    import importlib, inspect
    
    #print(sys.modules)
    

    for name in x.rglob("*.py"):
        print("###")
        # print(get_functions_and_methods(name))
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("module.name", "C:/Users/PD/Documents/MFF/grs-user-studies/server/plugins/fastcompare/algo/wrappers/lenskit.py")
        foo = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = foo
        
        spec.loader.exec_module(foo)
        for name, cls in inspect.getmembers(importlib.import_module("module.name"), inspect.isclass):
            print(name, cls)
        print(AlgorithmBase.__subclasses__())
        break
        print("##")
        
        # if name.resolve().absolute() != Path(__file__):
        #     print(f"Name stem = {name}")
        #     for n, cls in inspect.getmembers(name.replace(".py", ""), inspect.isclass):
        #         print(n, cls, issubclass(cls, AlgorithmBase))

    assert False

    # Find all python scripts from plugin directory (recursively)
    for name in x.rglob("*.py"):
        print(f"name = {name.stem}")
        for name, cls in inspect.getmembers(importlib.import_module("algorithm_base"), inspect.isclass):
            print(name, cls, issubclass(cls, AlgorithmBase), issubclass(cls, AlgorithmBase))