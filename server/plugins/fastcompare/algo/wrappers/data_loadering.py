import os
import time

import pandas as pd
import requests

from plugins.fastcompare.algo.algorithm_base import DataLoaderBase
from plugins.utils.ml_data_loader import MLDataLoader, RatingLowFilter, MovieFilterByYear, RatingFilterOld, RatingsPerYearFilter, RatingUserFilter, RatedMovieFilter, LinkFilter

from common import load_system_config

from app import pm
from PIL import Image
from io import BytesIO

# A wrapper against MLDataLoader from utils plugin that satisfy the DataLoaderBase interface
# The interface itself is specific to fastcompare so that is why 

class MLDataLoaderWrapper(DataLoaderBase):

    def __init__(self, **kwargs):

        datasets_base_dir = load_system_config()["datasets_base_path"]

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

        self.loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            [RatingLowFilter(4.0), MovieFilterByYear(1990), RatingFilterOld(2010), RatingsPerYearFilter(50.0), RatingUserFilter(100), RatedMovieFilter(), LinkFilter()],
            rating_matrix_path=None, img_dir_path=img_dir_path
        )

    def load_data(self):
        #cache_path = os.path.join(basedir, "static", "ml-latest", "data_cache.pckl")
        start_time = time.perf_counter()
        self.loader.load()
        print(f"## Loading took: {time.perf_counter() - start_time}")
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
    


class GoodbooksDataLoader(DataLoaderBase):

    def __init__(self, **kwargs):

        datasets_base_dir = load_system_config()["datasets_base_path"]

        if not os.path.exists(datasets_base_dir):
            assert False, f"Datasets base dir ({datasets_base_dir}) does not exist"
        
        if not os.path.exists(os.path.join(datasets_base_dir, "ml-latest")):
            assert False, f"ml-latest dataset is missing in the dataset base directory ({datasets_base_dir})"

        for f in ["ratings.csv", "movies.csv", "tags.csv", "links.csv", "img"]:
            if not os.path.exists(os.path.join(datasets_base_dir, "ml-latest", f)):
                assert False, f"{f} is missing in the {os.path.join(datasets_base_dir, 'ml-latest')} directory"

        self.ratings_path = os.path.join(datasets_base_dir, "goodbooks-10k", "ratings.csv")
        self.books_path = os.path.join(datasets_base_dir, "goodbooks-10k", "books.csv")
        self.tags_path = os.path.join(datasets_base_dir, "goodbooks-10k", "tags.csv")
        self.book_tags_path = os.path.join(datasets_base_dir, "goodbooks-10k", "book_tags.csv")
        self.img_dir_path = os.path.join(datasets_base_dir, "goodbooks-10k", "img")

        

    def load_data(self):
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
        

        already_downloaded = os.listdir(self.img_dir_path)
        self.book_index_to_url = dict()
        for img_name in already_downloaded:
            book_id = int(img_name.split(".jpg")[0])
            if book_id in self.book_id_to_index: # It could be possible we have more images than books (due to filtering)
                self.book_index_to_url[self.book_id_to_index[book_id]] = os.path.join(self.img_dir_path, img_name)

        print(f"## Loading took: {time.perf_counter() - start_time}")
        self._ratings_df = self._ratings_df.rename(columns={"user_id": "user", "book_id": "item_id"})
        self._ratings_df.loc[:, "item"] = self._ratings_df.item_id.map(lambda x: self.book_id_to_index[x])
        # TODO remove "movie" in column names to "item"

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

    def get_item_id_image_url(self, item_id):
        return self.get_item_index_image_url(self.get_item_index(item_id))
    
    def get_item_index_image_url(self, item_index):
        if self.img_dir_path:
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
                    img = img.resize((TARGET_WIDTH, new_height), Image.ANTIALIAS)
                    img.save(os.path.join(self.img_dir_path, f'{book_id}.jpg'), quality=90)

            # Use local version of images
            print(f"Norm path = {os.path.normpath(self.img_dir_path)}")
            suffix = os.path.normpath(self.img_dir_path).split(f"{os.sep}static{os.sep}")[1]
            p = os.path.normpath(os.path.join(suffix, f'{self.book_index_to_id[item_index]}.jpg'))
            return pm.emit_assets('utils', p.replace(os.sep, '/'))

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