import ast
import math
import os
from pathlib import Path
from pprint import pprint
import time
import random
from typing import List

import pandas as pd
import requests

from plugins.fastcompare.algo.algorithm_base import DataLoaderBase
from common import get_abs_project_root_path

# from algorithm_base import DataLoaderBase
# from common import get_abs_project_root_path

from flask import url_for, has_app_context
from PIL import Image
from io import BytesIO

class FMADataLoader(DataLoaderBase):

    PARETO_PARAM = 1.16
    USER_COUNT  = 100
    SEED = 'NEVERGONNAGIVEYOUUP'

    def __init__(self, **kwargs):
        datasets_base_dir: Path = get_abs_project_root_path() / 'static' / 'datasets'

        # Assert that source data exist
        assert datasets_base_dir.exists(), f"Dataset's base dir ({datasets_base_dir}) does not exist"
        assert (datasets_base_dir / 'fma').exists(), f"FMA dataset is missing in the dataset base directory ({datasets_base_dir})"

        for f in ["echonest.csv", "features.csv", "genres.csv", "tracks.csv"]:
            assert (datasets_base_dir / 'fma' / f).exists(), f"{f} is missing in the {datasets_base_dir / 'fma'} directory"

        self.echonest_path  = datasets_base_dir / "fma" / "echonest.csv"
        self.features_path  = datasets_base_dir / "fma" / "features.csv"
        self.genres_path    = datasets_base_dir / "fma" / "genres.csv"
        self.tracks_path    = datasets_base_dir / "fma" / "tracks.csv"
        self.img_dir_path   = datasets_base_dir / "fma" / "img"

        # Columns from tracks csv
        self._track_columns = {
            'item_id' :                     ('track', 'id'),
            'track_duration' :              ('track', 'duration'),
            'track_language' :              ('track', 'language_code'),
            'track_lyricist' :              ('track', 'lyricist'),
            'track_publisher' :             ('track', 'publisher'),
            'track_tags' :                  ('track', 'tags'),
            'title' :                       ('track', 'title'),
            'track_information' :           ('track', 'information'),
            'track_genre_top' :             ('track', 'genre_top'),
            'track_listens' :               ('track', 'listens'),
            #'track_genres_raw' :            ('track', 'genres_all'),
            'album_id' :                    ('album', 'id'),
            'album_title' :                 ('album', 'title'),
            'album_date_released' :         ('album', 'date_released'),
            'album_type' :                  ('album', 'type'),
            'artist_id' :                   ('artist', 'id'),
            'artist_name' :                 ('artist', 'name'),
            'artist_active_year_begin' :    ('artist', 'active_year_begin'),
            'artist_active_year_end' :      ('artist', 'active_year_end'),
            'artist_bio' :                  ('artist', 'bio'),
            'artist_location' :             ('artist', 'location'),
        }

        # Columns from echonest csv
        self._echonest_columns = {
            "acousticness" : ("echonest", "audio_features", "acousticness" ),
            "danceability" : ("echonest", "audio_features", "danceability" ),
            "energy" : ("echonest", "audio_features", "energy" ),
            "instrumentalness" : ("echonest", "audio_features", "instrumentalness" ),
            "liveness" : ("echonest", "audio_features", "liveness" ),
            "speechiness" : ("echonest", "audio_features", "speechiness" ),
            "tempo" : ("echonest", "audio_features", "tempo" ),
            "valence": ("echonest", "audio_features", "valence" ),
        }

        # Ensure img dir path exists
        self.img_dir_path.mkdir(parents=True, exist_ok=True)

    def construct_user_feedback(self, track_df: pd.DataFrame, user_count: int, parteo_param, seed):
        """Given track ids and listens dataframe, generate feedback for number of users."""

        print("Constructing user feedback")

        rng = random.Random(seed)

        ratings = []
        user_ids = list(range(user_count))

        # Create a list of user_count normalized pareto distributed values
            # pareto better mimics the general user behavior
        pareto_vars = [rng.paretovariate(parteo_param) for __ in range(user_count)]
        pareto_sum = sum(pareto_vars)
        norm_pareto_vars = [var/pareto_sum for var in pareto_vars]

        for __, row in track_df.iterrows():

            rng.shuffle(user_ids)
            total_listens = row['listens']
            track_id = row['track_id']
            overflow = 0.0

            for i, user_id in enumerate(user_ids):
                # Track listens must be integer values - rounded
                    # in order to not loose many listens on rounding, we are keeping 'overflow' 
                    # when it reaches 1, we increment the next listen count
                listens = total_listens * norm_pareto_vars[i]
                listens_rounded = math.floor(listens)
                overflow += listens - listens_rounded
                if overflow >= 1:
                    listens_rounded += 1
                    overflow -= 1

                ratings.append((user_id, track_id, listens_rounded))


        return pd.DataFrame(
            data = ratings,
            columns = ['user', 'item_id', 'rating']
        )
    
    def genre_ids_to_names(self, row: pd.Series) -> List[str]:
        """Translate genre ids to names."""

        processed_ids = eval(row[('track', 'genres_all')])
        processed_names = [self._genres_df.loc[id]["title"] for id in processed_ids]
        names_joined = '|'.join(processed_names)
        return names_joined


    def load_data(self):
        start_time = time.perf_counter()

        # Load sources
        self._echonest_df   = pd.read_csv(self.echonest_path, index_col=0, header=[0, 1, 2])
        self._genres_df     = pd.read_csv(self.genres_path, index_col=0)
        self._tracks_df     = pd.read_csv(self.tracks_path , index_col=0, header=[0, 1])

        # Preprocess tracks dataframe
        COLUMNS = [('album', 'date_released'),
                   ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            self._tracks_df [column] = pd.to_datetime(self._tracks_df [column])

        COLUMNS = [ ('album', 'type'), ('artist', 'bio')]
        for column in COLUMNS:
            self._tracks_df [column] = self._tracks_df [column].astype('category')


        # Extract columns from tracks
        tracks_columns_to_drop = set(self._tracks_df.columns.to_list()).difference(set(self._track_columns.values()))
        cleaned_tracks = self._tracks_df.drop(columns=tracks_columns_to_drop)
        cleaned_tracks.columns = ['_'.join(col) for col in cleaned_tracks.columns]
        cleaned_tracks.rename(columns={'track_title': 'title'}, inplace=True)

        # Extract columns from echonest
        echonest_columns_to_drop = set(self._echonest_df.columns.to_list()).difference(set(self._echonest_columns.values()))
        self._echonest_df.drop(columns=echonest_columns_to_drop, inplace=True)
        self._echonest_df.columns = [f'echonest_{col[-1]}' for col in self._echonest_df.columns]
        
        # Merge them
        self._items_indexed: pd.DataFrame = cleaned_tracks.join(self._echonest_df, how='left')
      
        # Translate genres
        self._items_indexed['genres'] = self._tracks_df.apply(lambda x: self.genre_ids_to_names(x), axis=1)

        # Create indexed/unindexed version
        self._items_indexed.index.rename('item_id', inplace=True)
        self._items = self._items_indexed.reset_index()

        # Add descriptions
        self._items.loc[:, "description"] = f"{self._items.title} by {self._items.artist_name}" # + ' ' + self.books_df.genres
        self.index_to_description = dict(zip(self._items.index, self._items.description))

        # index <=> id translation tables
        self.index_to_id = pd.Series(self._items.item_id.values,index=self._items.index).to_dict()
        self.id_to_index = pd.Series(self._items.index,index=self._items.item_id.values).to_dict()

        
        # Construct user fedback

        self._ratings_df = self.construct_user_feedback(
            pd.DataFrame(
                data = {
                    'track_id' : self._items_indexed.index,
                    'listens'  : self._items_indexed['track_listens'].copy()   
                }
            ),
            self.USER_COUNT,
            self.PARETO_PARAM,
            self.SEED
        )

        self._ratings_df.loc[:, "item"] = self._ratings_df.item_id.map(lambda x: self.id_to_index[x])


        # We needed listens only to construct ratings, so drop them
        self._items.drop(['track_listens'], axis=1)
        self._items_indexed.drop(['track_listens'], axis=1)


        print(f"## Loading took: {time.perf_counter() - start_time} seconds")


    # Returns dataframe with the interactions/ratings (be aware that implicit feedback is now considered)
    # There should be "user" and "item" columns in the dataframe
    @property
    def ratings_df(self):
        return self._ratings_df

    @property
    def items_df(self):
        return self._items

    @property
    def items_df_indexed(self):
        return self._items_indexed
    ###

    def get_item_id_image_url(self, item_id):
        print(f'id  = {item_id}')
        return self.get_item_index_image_url(self.get_item_index(item_id))
    
    def get_item_index_image_url(self, item_index):

        item_record: pd.Series = self.items_df.loc[item_index]
        album_id  = item_record.loc['album_id']
        artist_id = item_record.loc['artist_id']

        # Do we have the album cover?
        try:
            result: Path = next(self.img_dir_path.glob(f'{album_id}.jpeg'))
            print(result.absolute())
            return url_for('static', filename=f'datasets/fma/img/{result.name}')
        except StopIteration:
        # Do we have the artist photo?
            try:
                result: Path = next(self.img_dir_path.glob(f'artist_{artist_id}.jpeg'))
                if result.exists():
                    return url_for('static', filename=f'datasets/fma/img/{result.name}')
            except StopIteration:
        # We don't have anything - fallback to unknown.jpg
                return url_for('static', filename='datasets/fma/img/unknown.jpg')
            

    def get_item_index(self, item_id):
        return self.id_to_index[item_id]

    def get_item_id(self, item_index):
        return self.index_to_id[item_index]

    def get_item_index_description(self, item_index):
        return self.index_to_description[item_index]

    def get_item_id_description(self, item_id):
        return self.get_item_index_description(self.get_item_index(item_id))

    @classmethod
    def name(self):
        return "FMA songs dataset"

    @classmethod
    def parameters(self):
        return []
    
