"""Script used for album covers.

To use it, initialize discogs client object (with required values) on line 32,
then on lines 60 and 94 fill out user agent. The API requires it to be filled.
"""

from pathlib import Path
from urllib.error import URLError
import pandas as pd
import discogs_client as dc
import requests
import imghdr
import os

from common import get_abs_project_root_path


def get_album_covers(tracks_path: Path, images_path: Path):

    # Construct list of tuples (album name, artist name)
    tracks_df = pd.read_csv(tracks_path , index_col=0, header=[0, 1])
    album_artist_df = pd.DataFrame(
        data = {
            'album_name'  : tracks_df[('album', 'title')].copy(),
            'album_id'  : tracks_df[('album', 'id')].copy(),
            'artist' : tracks_df[('artist', 'name')].copy(),
            'artist_id' : tracks_df[('artist', 'id')].copy(),
        },
    ).drop_duplicates()


    tmp_path = images_path / 'tmp'
    tmp_path.mkdir(exist_ok=True, parents=True)


    client = dc.Client(
        '<ADD_USER_AGENT>',
        user_token='<ADD_DISCOGS_USER_TOKEN>'
    )


    for __, row in album_artist_df.iterrows():

        # Album cover is already downloaded
        if len(list(images_path.glob(f"{row['album_id']}.*"))) > 0:
            # print('exists') 
            continue

        print(f"Downloading: {row['album_name']} by {row['artist']}")
        
        # Download
        try:
            results = client.search(
                row['album_name'],
                type='release',
                artist=row['artist'],
                per_page=1,
                page=1
            )

        
            thumbnail_url = results[0].thumb
            response = requests.get(thumbnail_url, headers={'user-agent': '<ADD-AGENT-NAME>'})

            if response.status_code != 200:
                raise URLError(f'Status code = {response.status_code}')
            
            extension = imghdr.what(file=None, h=response.content)
            save_path = f"{row['album_id']}.{extension}"
            with open(images_path/save_path, 'wb') as f:
                f.write(response.content)

        except Exception as e :
            print(f"Download failed because [{type(e)}]: {e}")
            print(f'Fallback to artist {row["artist"]}')
        
            save_path = f"{row['album_id']}.none"
            (images_path/save_path).touch()

            if len(list(images_path.glob(f"artist_{row['artist_id']}.*"))) > 0:
                continue

            try:
                results = client.search(
                    row['artist'],
                    type='artist',
                    per_page=1,
                    page=1
                )
            
                image = results[0].images[0]
                if 'uri150' in image:
                    thumbnail_url = results[0].images[0]['uri150']
                else:
                    thumbnail_url = results[0].images[0]['resource_url']

                response = requests.get(thumbnail_url, headers={'user-agent': '<ADD_USER_AGENT>'})

                if response.status_code != 200:
                    raise URLError(f'Status code = {response.status_code}')
                
                extension = imghdr.what(file=None, h=response.content)
                save_path = f"artist_{row['artist_id']}.{extension}"
                with open(images_path/save_path, 'wb') as f:
                    f.write(response.content)

            except Exception as e :
                print(f"Fallback failed because [{type(e)}]: {e}")
                save_path = f"artist_{row['artist_id']}.none"
                (images_path/save_path).touch()



if __name__ == '__main__':
    dataset_path = os.path.join(get_abs_project_root_path(), 'static', 'datasets')
    img_data_path = Path(f'{dataset_path}/fma/img')
    tracks_path = Path(f'{dataset_path}/fma/tracks.csv')
    get_album_covers(tracks_path, img_data_path)

