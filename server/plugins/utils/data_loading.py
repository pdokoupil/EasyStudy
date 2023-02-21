import functools
import os
import pickle
import time
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from ml_data_loader import MLDataLoader, RatingUserFilter, RatingMovieFilter, RatedMovieFilter, TagsFilter, TagsRatedMoviesFilter, RatingTagFilter, MovieFilterByYear, RatingFilterOld, RatingsPerYearFilter, RatingLowFilter, LinkFilter
from composed_func import ComposedFunc
from rating_matrix_transform import SubtractMeanNormalize

basedir = os.path.abspath(os.path.dirname(__file__))

# Loads the movielens dataset
@functools.lru_cache(maxsize=None)
def load_ml_dataset(ml_variant="ml-latest"):
    cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
    if os.path.exists(cache_path):
        print(f"Trying to load data cache from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        print("Cache not available, loading everything again")
        
        ratings_path = os.path.join(basedir, "static", f"{ml_variant}/ratings.csv")
        movies_path = os.path.join(basedir, "static", f"{ml_variant}/movies.csv")
        rating_matrix_path = os.path.join(basedir, "static", f"{ml_variant}/rating_matrix.npy")
        tags_path = os.path.join(basedir, "static", f"{ml_variant}/tags.csv")
        links_path = os.path.join(basedir, "static", f"{ml_variant}/links.csv")
        img_dir_path = os.path.join(basedir, "static", "ml-latest", "img")
        
        start_time = time.perf_counter()
        loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            [RatingLowFilter(4.0), MovieFilterByYear(1990), RatingFilterOld(2010), RatingsPerYearFilter(50.0), RatingUserFilter(100), RatedMovieFilter(), LinkFilter()],
            rating_matrix_path=rating_matrix_path, img_dir_path=img_dir_path
        )
        loader.load()
        print(f"## Loading took: {time.perf_counter() - start_time}")

        print(f"Caching the data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(loader, f)

        return loader