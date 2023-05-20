import sys
import os

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from plugins.utils.preference_elicitation import  load_ml_dataset

import pickle

if __name__ == "__main__":
    loader = load_ml_dataset()
    
    with open("./ratings_df_preprocessed.pkl", "wb") as f:
        pickle.dump(loader.ratings_df, f)

    with open("./movie_index_to_id.pkl", "wb") as f:
        pickle.dump(loader.movie_index_to_id, f)

    with open("./movie_id_to_index.pkl", "wb") as f:
        pickle.dump(loader.movie_id_to_index, f) 