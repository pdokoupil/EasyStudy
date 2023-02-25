import functools
import os
import pickle
import time
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from ml_data_loader import MLDataLoader, RatingUserFilter, RatedMovieFilter, MovieFilterByYear, RatingFilterOld, RatingsPerYearFilter, RatingLowFilter, LinkFilter

from common import get_abs_project_root_path
from pathlib import Path

# Loads the movielens dataset
@functools.lru_cache(maxsize=None)
def load_ml_dataset(ml_variant="ml-latest"):
    basedir = os.path.join(get_abs_project_root_path(), 'static', 'datasets')
    #cache_base_dir = os.path.join(Path(__file__).parent.absolute(), "cache", 'utils', ml_variant)
    cache_base_dir = os.path.join(get_abs_project_root_path(), "cache", 'utils', ml_variant)
    Path(cache_base_dir).mkdir(parents=True, exist_ok=True)

    cache_path = os.path.join(cache_base_dir, "data_cache.pckl")
    if os.path.exists(cache_path):
        print(f"Trying to load data cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Cache not available, loading everything again")
        
        ratings_path = os.path.join(basedir, f"{ml_variant}/ratings.csv")
        movies_path = os.path.join(basedir, f"{ml_variant}/movies.csv")
        tags_path = os.path.join(basedir, f"{ml_variant}/tags.csv")
        links_path = os.path.join(basedir, f"{ml_variant}/links.csv")
        img_dir_path = os.path.join(basedir, ml_variant, "img")
        # Ensure img dir path exists
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            [RatingLowFilter(4.0), MovieFilterByYear(1990), RatingFilterOld(2010), RatingsPerYearFilter(50.0), RatingUserFilter(100), RatedMovieFilter(), LinkFilter()],
            img_dir_path=img_dir_path
        )
        loader.load()
        print(f"## Loading took: {time.perf_counter() - start_time}")

        print(f"Caching the data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(loader, f)

        return loader