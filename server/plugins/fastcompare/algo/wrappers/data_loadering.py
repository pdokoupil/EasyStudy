import json
import os
from pathlib import Path
import pickle
import time
import numpy as np

import pandas as pd
import requests

from cachetools import cached

from plugins.fastcompare.algo.algorithm_base import DataLoaderBase
from plugins.utils.ml_data_loader import MLDataLoader, RatingLowFilter, MovieFilterByYear, RatingFilterOld, RatingsPerYearFilter, RatingUserFilter, RatedMovieFilter, LinkFilter

from common import get_abs_project_root_path

from flask import url_for, has_app_context
from PIL import Image
from io import BytesIO

import multiprocessing
import tqdm
from imdb import Cinemagoer

from plugins.utils.helpers import cos_sim_np


N_PROCESSES = 6


# A wrapper against MLDataLoader from utils plugin that satisfy the DataLoaderBase interface
# The interface itself is specific to fastcompare so that is why 

class MLDataLoaderWrapper(DataLoaderBase):

    def __init__(self, **kwargs):

        datasets_base_dir = os.path.join(get_abs_project_root_path(), 'static', 'datasets')

        if not os.path.exists(datasets_base_dir):
            assert False, f"Datasets base dir ({datasets_base_dir}) does not exist"
        
        if not os.path.exists(os.path.join(datasets_base_dir, "ml-latest")):
            assert False, f"ml-latest dataset is missing in the dataset base directory ({datasets_base_dir})"

        for f in ["ratings.csv", "movies.csv", "tags.csv", "links.csv", "img"]:
            if not os.path.exists(os.path.join(datasets_base_dir, "ml-latest", f)):
                assert False, f"{f} is missing in the {os.path.join(datasets_base_dir, 'ml-latest')} directory"

        ratings_path = os.path.join(datasets_base_dir, "ml-latest", "ratings.csv")
        movies_path = os.path.join(datasets_base_dir, "ml-latest", "movies.csv")
        tags_path = os.path.join(datasets_base_dir, "ml-latest", "tags.csv")
        links_path = os.path.join(datasets_base_dir, "ml-latest", "links.csv")
        img_dir_path = os.path.join(datasets_base_dir, "ml-latest", "img")
        # Ensure img dir path exists
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        self.loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            [RatingLowFilter(4.0), MovieFilterByYear(1990), RatingFilterOld(2010), RatingsPerYearFilter(50.0), RatingUserFilter(100), RatedMovieFilter(), LinkFilter()],
            rating_matrix_path=None, img_dir_path=img_dir_path
        )

    def load_data(self, *args, **kwargs):
        #cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
        start_time = time.perf_counter()
        self.loader = self.loader.load()
        print(f"## Loading took: {time.perf_counter() - start_time}")
        
        self.loader.movies_df = self.loader.movies_df.rename(columns={"movieId": "item_id"})
        self.loader.movies_df_indexed = self.loader.movies_df_indexed.rename(columns={"movieId": "item_id"})

        self._ratings_df = self.loader.ratings_df.rename(columns={"userId": "user", "movieId": "item_id"})
        self._ratings_df.loc[:, "item"] = self._ratings_df.item_id.map(lambda x: self.loader.movie_id_to_index[x])
        # TODO remove "movie" in column names to "item"

    # Returns dataframe with the interactions/ratings (be aware that implicit feedback is now considered)
    # There should be "user" and "item" columns in the dataframe
    @property
    def ratings_df(self):
        return self._ratings_df

    @property
    def items_df(self):
        return self.loader.movies_df

    @property
    def items_df_indexed(self):
        return self.loader.movies_df_indexed
    
    @property
    def distance_matrix(self):
        return self.loader.distance_matrix
    
    @property
    def rating_matrix(self):
        return self.loader.rating_matrix

    def get_item_id_image_url(self, item_id):
        return self.loader.get_image(self.get_item_index(item_id))
    
    def get_item_index_image_url(self, item_index):
        return self.loader.get_image(item_index)

    def get_item_index(self, item_id):
        return self.loader.movie_id_to_index[item_id]

    def get_item_id(self, item_index):
        return self.loader.movie_index_to_id[item_index]

    def get_item_index_description(self, item_index):
        return self.loader.movie_index_to_description[item_index]

    def get_item_id_description(self, item_id):
        return self.get_item_index_description(self.get_item_index(item_id))

    @classmethod
    def name(self):
        return "Filtered ML-25M dataset"

    @classmethod
    def parameters(self):
        return []
    

class MLGenomeDataLoader(DataLoaderBase):

    MIN_MOVIE_RATINGS = 10

    def __init__(self, **kwargs):
        datasets_base_dir = os.path.join(get_abs_project_root_path(), 'static', 'datasets')

        if not os.path.exists(datasets_base_dir):
            assert False, f"Datasets base dir ({datasets_base_dir}) does not exist"
        
        if not os.path.exists(os.path.join(datasets_base_dir, "ml-genome-2021")):
            assert False, f"ml-genome-2021 dataset is missing in the dataset base directory ({datasets_base_dir})"

        for f in ["ratings.json", "metadata_updated.json"]:
            if not os.path.exists(os.path.join(datasets_base_dir, "ml-genome-2021", f)):
                assert False, f"{f} is missing in the {os.path.join(datasets_base_dir, 'ml-genome-2021')} directory"

        self.ratings_path = os.path.join(datasets_base_dir, "ml-genome-2021", "ratings.json")
        self.metadata_path = os.path.join(datasets_base_dir, "ml-genome-2021", "metadata_updated.json")
        self.img_dir_path = os.path.join(datasets_base_dir, "ml-genome-2021", "img")
        self.movie_data_path = os.path.join(datasets_base_dir, "ml-genome-2021", "movie_data_small.json")
        # Ensure img dir path exists
        Path(self.img_dir_path).mkdir(parents=True, exist_ok=True)

    def _get_imdb_subset(self, dat):
        
        # Try to find URL of a cover with largest resolution
        if dat.has_key("full-size cover url"):
            url = dat["full-size cover url"]
        elif dat.has_key("cover url"):
            url = dat["cover url"]
        elif dat.has_key("cover"):
            url = dat["cover"]
        else:
            url = None

        return {
                "plot": dat["plot"]  if "plot" in dat else [],
                "cast": [x["name"] for x in dat["cast"]] if "cast" in dat else [],
                "genres": dat["genres"] if "genres" in dat else [],
                "rating": dat["rating"] if "rating" in dat else -1,
                "year": dat["year"] if "year" in dat else -1,
                "cover": url
        }

    def _download_movie_data(self, unique_item_ids):
        print("Preparing download")
        #unique_item_ids = list(self.item_id_to_index.keys()) #df_merged_new.item_id.unique()
        unique_id_pairs = []
        for i in unique_item_ids:
            imdb_id = self.metadata_df_indexed.loc[i].imdbId
            unique_id_pairs.append((i, imdb_id))

        access = Cinemagoer()



        def get_imdb_info(id_pair):
            item_id, imdb_id = id_pair
            
            reps = 0
            while reps < 2:
                try:
                    res = self._get_imdb_subset(access.get_movie(imdb_id))
                    break
                except Exception as e:
                    print(f"Error for {imdb_id}: {e}")
                    d = 1.0 + 3 * np.random.random()
                    print(f"Adding random delay of: {d} seconds")
                    time.sleep(d)
                    reps += 1
                    res = None
                
            
            return item_id, imdb_id, res

        pool = multiprocessing.Pool(processes=N_PROCESSES)
        results = dict()

        start_time = time.perf_counter()
        print("Starting download")
        for r in tqdm.tqdm(pool.imap_unordered(get_imdb_info, unique_id_pairs)):
            if r[2] is not None:
                results[r[0]] = r[2]
        print(time.perf_counter() - start_time)
        return results

    def _download_images(self, movie_data):
        print(f"Starting downloading the images")
        def download_covers(x):
            item_id, res = x
            img_path = os.path.join(self.img_dir_path, f'{item_id}.jpg')
            # If the image already exists we can skip it
            if os.path.exists(img_path):
                return
            # If we do not have corresponding image data (from IMDb) we skip
            # since we do not have cover url
            if res is None:
                return

            url = res["cover"]
            if not url:
                return

            # Try downloading the image from cover
            # Allow for few failures so repeat it multiple times
            # to prevent issues with networking
            self._try_download_and_resize(url, img_path)

        pool = multiprocessing.Pool(processes=N_PROCESSES)

        start_time = time.perf_counter()
        for _ in tqdm.tqdm(pool.imap_unordered(download_covers, movie_data.items())):
            pass
        print(time.perf_counter() - start_time)
        print("Done downloading the images")

    def _try_download_and_resize(self, url, target_path):
        reps = 0
        while reps < 2:
            try:
                response = requests.get(url, stream=True)
                if not response.ok:
                    reps += 1
                    continue
                img = Image.open(BytesIO(response.content))
                width, height = img.size
                TARGET_WIDTH = 200
                coef = TARGET_WIDTH / width
                new_height = int(height * coef)
                img = img.resize((TARGET_WIDTH, new_height), Image.ANTIALIAS).convert('RGB')
                img.save(target_path, quality=90)
                return True
            except Exception as e:
                print(f"Error for {target_path}: {e}")
                d = 1.0 + 3 * np.random.random()
                print(f"Adding random delay of: {d} seconds")
                time.sleep(d)
                reps += 1

        return False


    def load_data(self, semi_local_cache_path, *args, **kwargs):

        if os.path.exists(semi_local_cache_path):
            print("There is already existing cache, loading from there")
            return self.load(None, None, semi_local_cache_path)

        #cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
        start_time = time.perf_counter()
        print("LOADER before loading files", time.strftime("%H:%M:%S", time.localtime()))
        self._ratings_df = pd.read_json(self.ratings_path, lines=True)
        print("LOADER loaded ratings", time.strftime("%H:%M:%S", time.localtime()))
        self.metadata_df = pd.read_json(self.metadata_path, lines=True)
        print("LOADER loaded metadata", time.strftime("%H:%M:%S", time.localtime()))

        ## Download IMDb data for all the movies ##
        self.metadata_df_indexed = self.metadata_df.set_index("item_id")
        if not os.path.exists(self.movie_data_path):
            self.movie_id_to_imdb_data = self._download_movie_data(self.metadata_df.item_id.unique())
            with open(self.movie_data_path, "w") as f:
                json.dump(self.movie_id_to_imdb_data, f, indent=4)
        else:
            print("Loading IMDb data")
            with open(self.movie_data_path, "r") as f:
                self.movie_id_to_imdb_data = json.load(f)
                self.movie_id_to_imdb_data = { int(k) : v for k, v in self.movie_id_to_imdb_data.items() }
        ## End of IMDb data handling ##
        print("LOADER below imdb handling", time.strftime("%H:%M:%S", time.localtime()))

        ## Handle images at the very beginning since we need to get rid of movies without image ##
        if not os.path.exists(self.img_dir_path) or len(os.listdir(self.img_dir_path)) == 0:
            # If there are no images at all, proceed to download
            self._download_images(self.movie_id_to_imdb_data)
        print("LOADER below img download", time.strftime("%H:%M:%S", time.localtime()))
        already_downloaded = [] if not os.path.exists(self.img_dir_path) else os.listdir(self.img_dir_path)
        self.movie_id_to_url = dict()
        has_image = set()
        self.all_genres = set()
        for img_name in already_downloaded:
            movie_id = int(img_name.split(".jpg")[0])
            has_image.add(movie_id)
            #if movie_id in self.item_id_to_index: # It could be possible we have more images than movies (due to filtering)
            self.movie_id_to_url[movie_id] = os.path.join(self.img_dir_path, img_name)
            self.all_genres.update(self.movie_id_to_imdb_data[movie_id]["genres"])

        ## End of Image handling ##
        print("LOADER below image handling", time.strftime("%H:%M:%S", time.localtime()))

        print("LOADER below imdb data mapping", time.strftime("%H:%M:%S", time.localtime()))

        # Filter out movies for which we do not have any images or IMDb data
        print(f"We have {self.metadata_df.item_id.unique().shape} movies now")
        self.metadata_df = self.metadata_df[(self.metadata_df.item_id.isin(has_image)) & (self.metadata_df.item_id.isin(self.movie_id_to_imdb_data))]
        print(f"We have {self.metadata_df.item_id.unique().shape} after filtering those without image/imdb data")
        self._ratings_df = self._ratings_df[self._ratings_df.item_id.isin(self.metadata_df.item_id.unique())]

        self._ratings_df = self._ratings_df.drop_duplicates(subset = ['user_id', 'item_id'], keep = 'last')
        print("LOADER below filtering 1", time.strftime("%H:%M:%S", time.localtime()))
        def parse_year(x):
            parts = x.split("(")
            if len(parts) != 2:
                return -1
            parts = parts[-1].split(")")
            if len(parts) != 2:
                return -1
            
            try:
                return int(parts[0])
            except:
                return -1

        self.metadata_df.loc[:, "year"] = self.metadata_df.title.map(parse_year)
        self.metadata_df = self.metadata_df[self.metadata_df.year != -1]
        print("LOADER below mapping", time.strftime("%H:%M:%S", time.localtime()))

        df_merged = self._ratings_df.merge(self.metadata_df, how="left", on="item_id")
        df_merged_filtered = df_merged[df_merged.title.notna()] # Filter out all ratings for movies for which we do not have metadata
        print("LOADER below filtering2", time.strftime("%H:%M:%S", time.localtime()))
        def full_stats(movie_year_thresh, user_year_thresh, df_mtd, df_mr):
            df_mtd = df_mtd[df_mtd.year >= movie_year_thresh]
            df_mr = df_mr[df_mr.item_id.isin(df_mtd.item_id)]
            passing_users = df_mr[df_mr.year >= user_year_thresh].user_id.unique()
            df_mr = df_mr[df_mr.user_id.isin(passing_users)]
            
            # Filter out movies that have < 10 ratings
            vcounts = df_mr.groupby("item_id").count()
            movies_with_enough_ratings = vcounts[vcounts.user_id >= MLGenomeDataLoader.MIN_MOVIE_RATINGS].index
            df_mr = df_mr[df_mr.item_id.isin(movies_with_enough_ratings)]
            
            print(f"Users: {df_mr.user_id.unique().size}, Items: {df_mr.item_id.unique().size}, Ratings: {df_mr.shape[0]}")
            return df_mtd, df_mr
        
        self.metadata_df, self._ratings_df = full_stats(1985, 2017, self.metadata_df, df_merged_filtered)
        self._ratings_df = self._ratings_df.drop_duplicates(subset=['item_id', 'user_id'], keep='last')

        rating_matrix_df = self._ratings_df.reset_index().pivot(index='user_id', columns='item_id', values="rating").fillna(0)
        self._rating_matrix = rating_matrix_df.values.astype(np.int8)
        print("LOADER below filtering3", time.strftime("%H:%M:%S", time.localtime()))

        self.item_id_to_index = {item_id : item_index for item_index, item_id in enumerate(rating_matrix_df.columns)}
        self.item_index_to_id = {item_index : item_id for item_index, item_id in enumerate(rating_matrix_df.columns)}
        print("LOADER below idx id mapping", time.strftime("%H:%M:%S", time.localtime()))
        # TODO REMOVE, was here just to verify we got same mapping as in GPU lab where we extracted features and relied on order
        assert self.item_index_to_id[147] == 181 and self.item_index_to_id[10000] == 119226

        print(f"## Loading took: {time.perf_counter() - start_time}")
        self._ratings_df = self._ratings_df.rename(columns={"user_id": "user"})
        self._ratings_df = self._ratings_df.drop(columns=["starring", "avgRating", "directedBy"]) # Drop unused columns to reduce size
        self._ratings_df.loc[:, "item"] = self._ratings_df.item_id.map(lambda x: self.item_id_to_index[x])
        print("LOADER below filtering4", time.strftime("%H:%M:%S", time.localtime()))
        # TODO remove "movie" in column names to "item"

        # Maps movie index to text description
        self.metadata_df.loc[:, "description"] = self.metadata_df.title
        self.metadata_df_indexed = self.metadata_df.set_index("item_id")
        self.item_index_to_description = { movie_idx : self.metadata_df_indexed.loc[movie_id].description for movie_idx, movie_id in self.item_index_to_id.items() }
        print("LOADER below filtering5", time.strftime("%H:%M:%S", time.localtime()))

        self._distance_matrix = 1.0 - cos_sim_np(self._rating_matrix.T)
        assert self._distance_matrix.dtype == np.float32 and self._rating_matrix.dtype == np.int8
    # Returns dataframe with the interactions/ratings (be aware that implicit feedback is now considered)
    # There should be "user" and "item" columns in the dataframe
    @property
    def ratings_df(self):
        return self._ratings_df

    @property
    def items_df(self):
        return self.metadata_df

    @property
    def items_df_indexed(self):
        return self.metadata_df_indexed

    @property
    def rating_matrix(self):
        return self._rating_matrix
    
    @property
    def distance_matrix(self):
        return self._distance_matrix
    
    @property
    def all_categories(self):
        return self.all_genres

    def get_item_id_image_url(self, item_id):
        return self.get_item_index_image_url(self.get_item_index(item_id))
    
    def get_item_index_image_url(self, item_index):
        item_id = self.item_index_to_id[item_index]
        if self.img_dir_path and has_app_context():
            assert item_id in self.movie_id_to_url, f"We expect to have images for all the movies"
            # Use local version of images
            return url_for('static', filename=f'datasets/ml-genome-2021/img/{item_id}.jpg')

        return self.movie_id_to_url[item_index]

    def get_item_index(self, item_id):
        return self.item_id_to_index[item_id]

    def get_item_id(self, item_index):
        return self.item_index_to_id[item_index]

    def get_item_index_description(self, item_index):
        return self.item_index_to_description[item_index]

    def get_item_id_description(self, item_id):
        return self.get_item_index_description(self.get_item_index(item_id))

    def get_item_index_categories(self, item_index):
        return self.movie_id_to_imdb_data[self.item_index_to_id[item_index]]["genres"]

    def get_item_id_categories(self, item_id):
        return self.movie_id_to_imdb_data[item_id]["genres"]

    def get_item_id_plot(self, item_id):
        if len(self.movie_id_to_imdb_data[item_id]["plot"]) > 0:
            return self.movie_id_to_imdb_data[item_id]["plot"][0] # Get the first of plots
        return ""

    @classmethod
    def name(self):
        return "Movielens Genome 2021 dataset"

    @classmethod
    def parameters(self):
        return []
    
    # We do the caching here, so that if multiple user studies are created, they can all share this instance assuming
    # they were created with same class_cache_path and semi_local_cache_path 
    @cached(cache={}, key=lambda data_loader, instance_cache_path, class_cache_path, semi_local_cache_name: f"{class_cache_path}/{semi_local_cache_name}")
    def load(self, instance_cache_path, class_cache_path, semi_local_cache_path):
        with open(semi_local_cache_path, "rb") as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        self.metadata_df_indexed = self.metadata_df.set_index("item_id")
        return self
        
        
    def save(self, instance_cache_path, class_cache_path, semi_local_cache_path):
        if os.path.exists(semi_local_cache_path):
            print("Not saving, cache already exists")
            return
        data = {
            "_ratings_df": self._ratings_df,
            "metadata_df": self.metadata_df,
            "img_dir_path": self.img_dir_path,
            "metadata_path": self.metadata_path,
            "ratings_path": self.ratings_path,
            "movie_data_path": self.movie_data_path,
            "movie_id_to_imdb_data": self.movie_id_to_imdb_data,
            "movie_id_to_url": self.movie_id_to_url,
            "all_genres": self.all_genres,
            "item_id_to_index": self.item_id_to_index,
            "item_index_to_id": self.item_index_to_id,
            "item_index_to_description": self.item_index_to_description,
            "_distance_matrix": self._distance_matrix,
            "_rating_matrix": self._rating_matrix
        }
        with open(semi_local_cache_path, "wb") as f:
            pickle.dump(data, f)


class GoodbooksDataLoader(DataLoaderBase):

    def __init__(self, **kwargs):
        datasets_base_dir = os.path.join(get_abs_project_root_path(), 'static', 'datasets')

        if not os.path.exists(datasets_base_dir):
            assert False, f"Datasets base dir ({datasets_base_dir}) does not exist"
        
        if not os.path.exists(os.path.join(datasets_base_dir, "goodbooks-10k")):
            assert False, f"goodbooks-10k dataset is missing in the dataset base directory ({datasets_base_dir})"

        for f in ["ratings.csv", "books.csv", "tags.csv", "book_tags.csv", "img"]:
            if not os.path.exists(os.path.join(datasets_base_dir, "goodbooks-10k", f)):
                assert False, f"{f} is missing in the {os.path.join(datasets_base_dir, 'goodbooks-10k')} directory"

        self.ratings_path = os.path.join(datasets_base_dir, "goodbooks-10k", "ratings.csv")
        self.books_path = os.path.join(datasets_base_dir, "goodbooks-10k", "books.csv")
        self.tags_path = os.path.join(datasets_base_dir, "goodbooks-10k", "tags.csv")
        self.book_tags_path = os.path.join(datasets_base_dir, "goodbooks-10k", "book_tags.csv")
        self.img_dir_path = os.path.join(datasets_base_dir, "goodbooks-10k", "img")
        # Ensure img dir path exists
        Path(self.img_dir_path).mkdir(parents=True, exist_ok=True)

    def load_data(self, *args, **kwargs):
        #cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
        start_time = time.perf_counter()

        self._ratings_df = pd.read_csv(self.ratings_path)
        self.books_df = pd.read_csv(self.books_path)
        self.tags_df = pd.read_csv(self.tags_path)
        self.book_tags_df = pd.read_csv(self.book_tags_path)

        self._ratings_df = self._ratings_df[self._ratings_df.rating >= 4]
        self._ratings_df = self._ratings_df[self._ratings_df.book_id.map(self._ratings_df.book_id.value_counts()) >= 400]
        self._ratings_df = self._ratings_df[self._ratings_df.user_id.map(self._ratings_df.user_id.value_counts()) >= 75]
        self._ratings_df = self._ratings_df.reset_index(drop=True)

        self.books_df = self.books_df[self.books_df.book_id.isin(self._ratings_df.book_id)]
        self.books_df = self.books_df.reset_index(drop=True)

        #self._ratings_df.merge(self.books_df[["book_id", "title"]])

        self.book_tags_df = pd.read_csv(self.book_tags_path)
        self.book_tags_df = self.book_tags_df.reset_index(drop=True)

        self.book_index_to_id = pd.Series(self.books_df.book_id.values,index=self.books_df.index).to_dict()
        self.book_id_to_index = pd.Series(self.books_df.index,index=self.books_df.book_id.values).to_dict()

        self.books_df = self.books_df.rename(columns={"book_id": "item_id"})
        self.books_df_indexed = self.books_df.set_index("item_id")
        
        self.books_df.loc[:, "description"] = self.books_df.title # + ' ' + self.books_df.genres
        self.book_index_to_description = dict(zip(self.books_df.index, self.books_df.description))
        

        already_downloaded = [] if not os.path.exists(self.img_dir_path) else os.listdir(self.img_dir_path)
        self.book_index_to_url = dict()
        for img_name in already_downloaded:
            book_id = int(img_name.split(".jpg")[0])
            if book_id in self.book_id_to_index: # It could be possible we have more images than books (due to filtering)
                self.book_index_to_url[self.book_id_to_index[book_id]] = os.path.join(self.img_dir_path, img_name)

        print(f"## Loading took: {time.perf_counter() - start_time}")
        self._ratings_df = self._ratings_df.rename(columns={"user_id": "user", "book_id": "item_id"})
        self._ratings_df.loc[:, "item"] = self._ratings_df.item_id.map(lambda x: self.book_id_to_index[x])
        # TODO remove "movie" in column names to "item"

        self._rating_matrix = self._ratings_df.reset_index().pivot(index='user_id', columns='item_id', values="rating").fillna(0).values
        self._distance_matrix = 1.0 - cos_sim_np(self._rating_matrix.T)

    # Returns dataframe with the interactions/ratings (be aware that implicit feedback is now considered)
    # There should be "user" and "item" columns in the dataframe
    @property
    def ratings_df(self):
        return self._ratings_df

    @property
    def items_df(self):
        return self.books_df

    @property
    def items_df_indexed(self):
        return self.books_df_indexed

    @property
    def rating_matrix(self):
        return self._rating_matrix
    
    @property
    def distance_matrix(self):
        return self._distance_matrix

    def get_item_id_image_url(self, item_id):
        return self.get_item_index_image_url(self.get_item_index(item_id))
    
    def get_item_index_image_url(self, item_index):
        if self.img_dir_path and has_app_context():
            if item_index not in self.book_index_to_url:
                # Download it first if it is missing
                book_id = self.book_index_to_id[item_index]
                remote_url = self.books_df_indexed.loc[book_id].image_url
                
                err = False
                try:
                    resp = requests.get(remote_url, stream=True)
                except:
                    err = True

                if err or resp.status_code != 200:
                    print(f"Failed download image for book with id={book_id}")
                else:
                    img = Image.open(BytesIO(resp.content))
                    width, height = img.size
                    TARGET_WIDTH = 200
                    coef = TARGET_WIDTH / width
                    new_height = int(height * coef)
                    img = img.resize((TARGET_WIDTH, new_height), Image.ANTIALIAS).convert('RGB')
                    img.save(os.path.join(self.img_dir_path, f'{book_id}.jpg'), quality=90)

            # Use local version of images
            item_id = self.book_index_to_id[item_index]
            return url_for('static', filename=f'datasets/goodbooks-10k/img/{item_id}.jpg')

        return self.book_index_to_url[item_index]

    def get_item_index(self, item_id):
        return self.book_id_to_index[item_id]

    def get_item_id(self, item_index):
        return self.book_index_to_id[item_index]

    def get_item_index_description(self, item_index):
        return self.book_index_to_description[item_index]

    def get_item_id_description(self, item_id):
        return self.get_item_index_description(self.get_item_index(item_id))

    @classmethod
    def name(self):
        return "Goodbooks-10k dataset"

    @classmethod
    def parameters(self):
        return []