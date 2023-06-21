# Postprocessing script for rating reconstruction

from datetime import datetime
import json
import pickle
import sys
import os
import time
import numpy as np
import scipy

import argparse

[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

import tensorflow as tf

from plugins.utils.preference_elicitation import prepare_tf_model, load_ml_dataset, prepare_wrapper, weighted_average_strategy, rlprop, weighted_average, recommend_2_3
from plugins.utils.rlprop_wrapper import get_supports

import pandas as pd

loader = load_ml_dataset()


import functools

from sklearn.metrics import ndcg_score

N_ITERATIONS = 8
K = 10

def predict_ratings(selected_movies, filter_out_movies = [], k = 10, return_model = False):
    

    max_user = loader.ratings_df.userId.max()

    ################ TF specific ################
    model, train = prepare_tf_model(loader)

    
    new_user = tf.constant(str(max_user + 1))
    def data_gen():
        for x in selected_movies:
            yield {
                "item_title": tf.constant(loader.movies_df.loc[x].title),
                "user_id": new_user,
            }
    ratings2 = tf.data.Dataset.from_generator(data_gen, output_signature={
        "item_title": tf.TensorSpec(shape=(), dtype=tf.string),
        "user_id": tf.TensorSpec(shape=(), dtype=tf.string)
    })

    # Finetune
    model.fit(ratings2.concatenate(train.take(100)).batch(256), epochs=2)

    seen_movies_tensor = tf.stack(
        [tf.constant(loader.movies_df.loc[x].title) for x in selected_movies]
        +
        [tf.constant(loader.movies_df.loc[x].title) for x in filter_out_movies]
    )
    
    scores, x = model.predict_all_unseen(new_user, seen_movies_tensor, n_items=loader.rating_matrix.shape[1]) #model.predict_for_user(new_user, ratings2, k=2000)
    scores, x = tf.squeeze(scores).numpy(), tf.squeeze(x).numpy()
    
    #scores = (scores - scores.min()) / (scores.max() - scores.min())
    print(f"Full shape = {scores.shape}")
    print(f"Scores = {scores[:k]}, x = {x[:k]}")

    if return_model:
        return x, scores, model

    return x, scores

def set_iteration(row):
    if row.interaction_type == "iteration-started" or row.interaction_type == "iteration-ended":
        row['iteration'] = json.loads(row.data)['iteration']
    else:
        row['iteration'] = None
    return row

def create_elicitation_selections_csv(df_interaction):
    df_result = pd.DataFrame(columns = [
        "userId",
        "movieId",
        "timestamp"
    ])
    
    df_elicitation_ends = df_interaction[df_interaction.interaction_type == "elicitation-ended"]
    for _, row in df_elicitation_ends.iterrows():
        d = json.loads(row.data)
        
        for movie_idx in d["elicitation_selected_movies"]:
            new_row = [
                row.participation,
                movie_idx,
                int(datetime.fromisoformat(row.time).timestamp())
            ]
            df_result = pd.concat([df_result, pd.DataFrame([new_row], columns=df_result.columns)], ignore_index=True)
                
    return df_result

def create_selections_csv(df_interaction):
    df_result = pd.DataFrame(columns = [
        "userId",
        "movieId",
        "timestamp",
        "session",
        "rank",
        "mo"
    ])
    
    df_selections = df_interaction[df_interaction.interaction_type == "selected-item"]
    for _, row in df_selections.iterrows():
        d = json.loads(row.data)
        new_row = [
            row.participation,
            d["selected_item"]["movie_idx"],
            int(datetime.fromisoformat(row.time).timestamp()),
            row.iteration,
            d["selected_item"]["rank"],
            d["selected_item"]["variant_name"] != "BETA"
        ]
        df_result = pd.concat([df_result, pd.DataFrame([new_row], columns=df_result.columns)], ignore_index=True)
    return df_result

def create_impressions_csv(df_interaction):
    df_result = pd.DataFrame(columns = [
        "userId",
        "movieId",
        "timestamp",
        "iteration",
        "rank"
    ])
    
    df_iteration_starts = df_interaction[df_interaction.interaction_type == "iteration-started"].drop_duplicates(subset=['participation', "iteration"], keep="last")
    for _, row in df_iteration_starts.iterrows():
        d = json.loads(row.data)
        
        actually_shown_algorithms = set()
        for x in d["algorithm_assignment"].values():
            if x["order"] >= 0:
                actually_shown_algorithms.add(x["name"])
        
        for algo, algo_data in d["movies"].items():
            if algo not in actually_shown_algorithms:
                continue

            for rank, movie in enumerate(algo_data["movies"]):
                new_row = [
                    row.participation,
                    movie["movie_idx"],
                    int(datetime.fromisoformat(row.time).timestamp()),
                    row.iteration,
                    rank
                ]
                df_result = pd.concat([df_result, pd.DataFrame([new_row], columns=df_result.columns)], ignore_index=True)
                
    return df_result

def create_beta_impressions_csv(df_interaction):
    df_result = pd.DataFrame(columns = [
        "userId",
        "movieId",
        "timestamp",
        "iteration",
        "rank"
    ])
    
    df_iteration_starts = df_interaction[df_interaction.interaction_type == "iteration-started"].drop_duplicates(subset=['participation', "iteration"], keep="last")
    for _, row in df_iteration_starts.iterrows():
        d = json.loads(row.data)
        
        actually_shown_algorithms = {"BETA"}
        for algo, algo_data in d["movies"].items():
            if algo not in actually_shown_algorithms:
                continue

            for rank, movie in enumerate(algo_data["movies"]):
                new_row = [
                    row.participation,
                    movie["movie_idx"],
                    int(datetime.fromisoformat(row.time).timestamp()),
                    row.iteration,
                    rank
                ]
                df_result = pd.concat([df_result, pd.DataFrame([new_row], columns=df_result.columns)], ignore_index=True)
                
    return df_result


def create_weights_csv(df_interaction):
    df_result = pd.DataFrame(columns = [
        "userId",
        "relevanceWeight",
        "diversityWeight",
        "noveltyWeight",
        "timestamp",
        "iteration"
    ])
    
    df_iteration_starts = df_interaction[df_interaction.interaction_type == "iteration-started"]
    for _, row in df_iteration_starts.iterrows():
        d = json.loads(row.data)
        
        shown_moo = None
        for x in d["algorithm_assignment"].values():
            if x["order"] >= 0 and x["name"] in {"DELTA", "GAMMA"}:
                shown_moo = x["name"]
                break
        
        [rel, div, nov] = d["weights"][shown_moo]
        
        new_row = [
            row.participation,
            rel,
            div,
            nov,
            int(datetime.fromisoformat(row.time).timestamp()),
            row.iteration
        ]
        df_result = pd.concat([df_result, pd.DataFrame([new_row], columns=df_result.columns)], ignore_index=True)
        
    return df_result

def calc_ratings(df_interaction, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx):
    df_ratings = pd.DataFrame(columns=[
        "userId",
        "movieId",
        "rawRating",
        "iteration"
    ])

    for _, row in df_interaction.iterrows():
        d = json.loads(row.data)
        participation = row.participation

        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()

        df_impressions_user = df_impressions[df_impressions.userId == participation]
        df_impressions_beta_user = df_impressions_beta[df_impressions_beta.userId == participation]

        for i in range(N_ITERATIONS):
            already_seen = df_impressions_user[df_impressions_user.iteration < i + 1].movieId.values.astype(int).tolist()
            filter_out_movies = elicitation_selected + already_seen
            selected_movies = elicitation_selected + sum(d["selected"][:i], [])

            movie_titles, scores = predict_ratings(selected_movies, filter_out_movies)
            movie_indices = [movie_title_to_idx[t] for t in movie_titles]
            
            df_ratings_user = pd.DataFrame({
                "userId": np.repeat(participation, scores.size),
                "movieId": movie_indices,
                "rawRating": scores,
                "session": np.repeat(i + 1, scores.size)    
            })
            
            #print(df_ratings_user.head())
            df_ratings = pd.concat([df_ratings, df_ratings_user], axis=0)
            #print("DF_ratings =")
            #print(df_ratings.head())

            #real = df_impressions_beta_user[df_impressions_beta_user.iteration == i + 1].movieId.values.astype(int).tolist()
            
            #print(f"@@ i={i}, real={real}, participation={participation}")
            #print(f"@@ predicted={x}, already_seen={already_seen}, elicitation_selected={elicitation_selected}, rest={sum(d['selected'][:i], [])}")
            
            
            #print(f"For iteration = {i + 1} we assume following recommendation: {x}, real was: {real}")
            #assert x == real
            # 

    return df_ratings


def calc_beta_supports(df_interaction, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx, df_weights):
    df_beta_supports = pd.DataFrame(columns=[
        "userId",
        "movieId",
        "relevanceSupport",
        "diversitySupport",
        "noveltySupport",
        "session",
        "timestamp",
        "rank"
    ])

    for _, row in df_interaction.iterrows():
        d = json.loads(row.data)
        participation = row.participation
        start_time = time.perf_counter()
        print(f"Participant = {participation} STARTING")

        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()

        df_impressions_user = df_impressions[df_impressions.userId == participation]
        df_impressions_beta_user = df_impressions_beta[df_impressions_beta.userId == participation]
        print(df_impressions_beta_user)

        for i in range(N_ITERATIONS):
            already_seen = df_impressions_user[df_impressions_user.iteration < i + 1].movieId.values.astype(int).tolist()
            filter_out_movies = elicitation_selected + already_seen
            selected_movies = elicitation_selected + sum(d["selected"][:i], [])

            movie_titles, scores, model = predict_ratings(selected_movies, filter_out_movies, return_model = True)
            movie_indices = [movie_title_to_idx[t] for t in movie_titles]
            
            #w = df_weights[(df_weights.userId == participation) & (df_weights.iteration == i + 1)].iloc[0]
            simulated_beta_result = weighted_average(selected_movies, model, np.array([1.0, 0.0, 0.0]), filter_out_movies, k=10, include_support=True)
            assert len(simulated_beta_result) == 10
            
            #weighted_average_res = weighted_average(selected_movies, model, np.array([w.relevanceWeight, w.diversityWeight, w.noveltyWeight]), filter_out_movies, k=10, include_support=True)
            #rlprop_res = rlprop(selected_movies, model, np.array([w.relevanceWeight, w.diversityWeight, w.noveltyWeight]), filter_out_movies, k=10, include_support=True)
            real = df_impressions_beta_user[df_impressions_beta_user.iteration == i + 1].movieId.values.astype(int).tolist()
            
            #print(f"iteration i = {i}, participation = {participation}")
            #print(f"Real BETA result = {real}")
            #print(f"Simulated BETA result with supports = {simulated_beta_result}")
            
            #print(f"Weighted average result = {weighted_average_res}")
            #print(f"Rlprop result = {rlprop_res}")
            
            if real != [int(x["movie_idx"]) for x in simulated_beta_result]: #, f"{real} != {[int(x['movie_idx']) for x in simulated_beta_result]}"
                print(f"Warning for participant = {participation}, and iteration == {i}, fallback calculation of supports")
                _, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies)
                wrapper.init()
                partial_list = []
                simulated_beta_result = []
                ### Support calculation ###
                users_partial_lists = np.full((wrapper.extended_rating_matrix.shape[0], len(real)), -1, dtype=np.int32)
                
                for idx, m_index in enumerate(real):
                    
                    # Calculate support values
                    supports = get_supports(users_partial_lists, wrapper.items,
                                                    wrapper.extended_rating_matrix, wrapper.distance_matrix,
                                                    wrapper.users_viewed_item, k=idx+1, n_users=wrapper.n_users)
                    
                    supports[0, :, :] = wrapper.normalizations[0](supports[0].T).T * wrapper.discount_sequences[0][idx]
                    supports[1, :, :] = wrapper.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[1][idx]
                    ### End of support calculation ###
                    simulated_beta_result.append({
                        "movie_idx": m_index,
                        "support": {
                            "relevance": supports[0, 0, idx],
                            "diversity": supports[1, 0, idx],
                            "novelty": supports[2, 0, idx]
                        },
                        "rank": idx
                    })
                    partial_list.append(m_index)

                assert partial_list == real, f"{partial_list} != {real}"

            
            for res in simulated_beta_result:
                df_supports_user = pd.DataFrame({
                    "userId": [participation],
                    "movieId": [int(res["movie_idx"])],
                    "relevanceSupport": [res["support"]["relevance"]],
                    "diversitySupport": [res["support"]["diversity"]],
                    "noveltySupport": [res["support"]["novelty"]],
                    "session": [i + 1],
                    "timestamp": [int(datetime.fromisoformat(row.time).timestamp())],
                    "rank": [res["rank"]]
                })
            
                df_beta_supports = pd.concat([df_beta_supports, df_supports_user], axis=0)


            #print(f"@@ i={i}, real={real}, participation={participation}")
            #print(f"@@ predicted={x}, already_seen={already_seen}, elicitation_selected={elicitation_selected}, rest={sum(d['selected'][:i], [])}")
            
            
            #print(f"For iteration = {i + 1} we assume following recommendation: {x}, real was: {real}")
            #assert x == real
            # 
        print(f"Participant = {participation} DONE after {time.perf_counter() - start_time}")
    return df_beta_supports


# Calculates supports w.r.t past user selections, not w.r.t. items shown from a single recommendation algorithm
# Only users selection during preference elicitation are considered for rating matrix estimation
def calc_adjusted_supports(df_interaction, df_elicitation, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx):
    df_supports = pd.DataFrame(columns=[
        "userId",
        "movieId",
        "relevanceSupport",
        "diversitySupport",
        "noveltySupport",
        "session",
        "rank",
        "algo",
        "mo"
    ])

    num_broken = 0

    for _, row in df_interaction.iterrows():
        d = json.loads(row.data)
        participation = row.participation
        
        print(f"Participant = {participation} STARTING")

        el_ended = df_elicitation[(df_elicitation.participation == participation)].iloc[0]
        orig_permutation = json.loads(el_ended.data)["orig_permutation"]

        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()

        #df_impressions_user = df_impressions[df_impressions.userId == participation]
        #df_impressions_beta_user = df_impressions_beta[df_impressions_beta.userId == participation]
        #print(df_impressions_beta_user)

        ### Support calculation ###
        #users_partial_lists = np.full((wrapper.extended_rating_matrix.shape[0], len(real)), -1, dtype=np.int32)
        movie_titles, scores, model = predict_ratings(elicitation_selected, elicitation_selected, return_model = True)
        selected_variants = d["selected_variants"]
        _, wrapper = prepare_wrapper(elicitation_selected, model, weighted_average_strategy, np.array([0.0, 0.5, 0.5]), elicitation_selected, k=len(sum(d["selected"], [])))
        wrapper.init()
        #partial_list = []
        result = []
        
        #selected_algorithms = d
        all_selected = []
        
        if len(selected_variants) != 8:
            num_broken += 1
            print(f"######### NUM BROKEN = {num_broken}")
            continue

        for iter, (selected_movie_idxs, selected_variants) in enumerate(zip(d["selected"], selected_variants)):

            for selected_movie_idx, selected_variant in zip(selected_movie_idxs, selected_variants):
                algo = None
                for algo_name, variant in orig_permutation[iter].items():
                    if variant == selected_variant:
                        algo = algo_name

                all_selected.append({
                    "movie_idx": selected_movie_idx,
                    "variant": selected_variant,
                    "algorithm": algo,
                    "mo": algo != "BETA",
                    "session": iter + 1
                })



        users_partial_lists = np.full((wrapper.extended_rating_matrix.shape[0], len(all_selected)), -1, dtype=np.int32)
        
        for idx, selected_info in enumerate(all_selected):
            
            # Calculate support values
            supports = get_supports(users_partial_lists, wrapper.items,
                                            wrapper.extended_rating_matrix, wrapper.distance_matrix,
                                            wrapper.users_viewed_item, k=idx+1, n_users=wrapper.n_users)
            
            supports[0, :, :] = wrapper.normalizations[0](supports[0].T).T * wrapper.discount_sequences[0][idx]
            supports[1, :, :] = wrapper.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[1][idx]
            supports[2, :, :] = wrapper.normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[2][idx]
            ### End of support calculation ###
            result.append({
                "movie_idx": selected_info["movie_idx"],
                "support": {
                    "relevance": supports[0, 0, idx],
                    "diversity": supports[1, 0, idx],
                    "novelty": supports[2, 0, idx]
                },
                "rank": idx,
                "mo": selected_info["mo"],
                "algorithm": selected_info["algorithm"]
            })

            new_row = [
                participation,
                selected_info["movie_idx"],
                supports[0, 0, idx],
                supports[1, 0, idx],
                supports[2, 0, idx],
                selected_info["session"],
                idx,
                selected_info["algorithm"],
                selected_info["mo"]
            ]
            df_supports = pd.concat([df_supports, pd.DataFrame([new_row], columns=df_supports.columns)], axis=0)

            users_partial_lists[0, idx] = selected_info["movie_idx"]


    #     for i in range(N_ITERATIONS):
    #         already_seen = df_impressions_user[df_impressions_user.iteration < i + 1].movieId.values.astype(int).tolist()
    #         filter_out_movies = elicitation_selected + already_seen
    #         selected_movies = elicitation_selected + sum(d["selected"][:i], [])

    #         movie_titles, scores, model = predict_ratings(selected_movies, filter_out_movies, return_model = True)
    #         movie_indices = [movie_title_to_idx[t] for t in movie_titles]
            
    #         #w = df_weights[(df_weights.userId == participation) & (df_weights.iteration == i + 1)].iloc[0]
    #         #simulated_beta_result = weighted_average(selected_movies, model, np.array([1.0, 0.0, 0.0]), filter_out_movies, k=10, include_support=True)
    #         #assert len(simulated_beta_result) == 10
            
    #         #weighted_average_res = weighted_average(selected_movies, model, np.array([w.relevanceWeight, w.diversityWeight, w.noveltyWeight]), filter_out_movies, k=10, include_support=True)
    #         #rlprop_res = rlprop(selected_movies, model, np.array([w.relevanceWeight, w.diversityWeight, w.noveltyWeight]), filter_out_movies, k=10, include_support=True)
    #         real = df_impressions_beta_user[df_impressions_beta_user.iteration == i + 1].movieId.values.astype(int).tolist()
            
    #         #print(f"iteration i = {i}, participation = {participation}")
    #         #print(f"Real BETA result = {real}")
    #         #print(f"Simulated BETA result with supports = {simulated_beta_result}")
            
    #         #print(f"Weighted average result = {weighted_average_res}")
    #         #print(f"Rlprop result = {rlprop_res}")
            
    #         #if real != [int(x["movie_idx"]) for x in simulated_beta_result]: #, f"{real} != {[int(x['movie_idx']) for x in simulated_beta_result]}"
            
    #         print(f"Warning for participant = {participation}, and iteration == {i}, fallback calculation of supports")
    #         _, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies)
    #         wrapper.init()
    #         partial_list = []
    #         simulated_beta_result = []
            
            
    #         for idx, m_index in enumerate(real):
                
    #             # Calculate support values
    #             supports = get_supports(users_partial_lists, wrapper.items,
    #                                             wrapper.extended_rating_matrix, wrapper.distance_matrix,
    #                                             wrapper.users_viewed_item, k=idx+1, n_users=wrapper.n_users)
                
    #             supports[0, :, :] = wrapper.normalizations[0](supports[0].T).T * wrapper.discount_sequences[0][idx]
    #             supports[1, :, :] = wrapper.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[1][idx]
    #             supports[2, :, :] = wrapper.normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[2][idx]
    #             ### End of support calculation ###
    #             simulated_beta_result.append({
    #                 "movie_idx": m_index,
    #                 "support": {
    #                     "relevance": supports[0, 0, idx],
    #                     "diversity": supports[1, 0, idx],
    #                     "novelty": supports[2, 0, idx]
    #                 },
    #                 "rank": idx
    #             })
    #             partial_list.append(m_index)

    #         assert partial_list == real, f"{partial_list} != {real}"

            
    #         for res in simulated_beta_result:
    #             df_supports_user = pd.DataFrame({
    #                 "userId": [participation],
    #                 "movieId": [int(res["movie_idx"])],
    #                 "relevanceSupport": [res["support"]["relevance"]],
    #                 "diversitySupport": [res["support"]["diversity"]],
    #                 "noveltySupport": [res["support"]["novelty"]],
    #                 "session": [i + 1],
    #                 "timestamp": [int(datetime.fromisoformat(row.time).timestamp())],
    #                 "rank": [res["rank"]]
    #             })
            
    #             df_beta_supports = pd.concat([df_beta_supports, df_supports_user], axis=0)


    #         #print(f"@@ i={i}, real={real}, participation={participation}")
    #         #print(f"@@ predicted={x}, already_seen={already_seen}, elicitation_selected={elicitation_selected}, rest={sum(d['selected'][:i], [])}")
            
            
    #         #print(f"For iteration = {i + 1} we assume following recommendation: {x}, real was: {real}")
    #         #assert x == real
    #         # 
    #     print(f"Participant = {participation} DONE after {time.perf_counter() - start_time}")
    # return df_beta_supports
    print(f"######### NUM BROKEN = {num_broken}")
    return df_supports

# Same as function above, but works iteratively, rating matrix is recaculated after every iteration
def calc_adjusted_supports_2(df_interaction, df_elicitation, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx):
    df_supports = pd.DataFrame(columns=[
        "userId",
        "movieId",
        "relevanceSupport",
        "diversitySupport",
        "noveltySupport",
        "session",
        "rank",
        "algo",
        "mo"
    ])

    num_broken = 0

    for _, row in df_interaction.iterrows():
        d = json.loads(row.data)
        participation = row.participation
        start_time = time.perf_counter()
        print(f"Participant = {participation} STARTING")

        
        el_ended = df_elicitation[(df_elicitation.participation == participation)].iloc[0]
        orig_permutation = json.loads(el_ended.data)["orig_permutation"]

        
        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()

        df_impressions_user = df_impressions[df_impressions.userId == participation]
        df_impressions_beta_user = df_impressions_beta[df_impressions_beta.userId == participation]
        # print(df_impressions_beta_user)

        selected_variants = d["selected_variants"]
        
        #selected_algorithms = d
        all_selected = []
        
        if len(selected_variants) != 8:
            num_broken += 1
            print(f"######### NUM BROKEN = {num_broken}")
            continue

        for iter, (selected_movie_idxs, selected_variants) in enumerate(zip(d["selected"], selected_variants)):

            for selected_movie_idx, selected_variant in zip(selected_movie_idxs, selected_variants):
                algo = None
                for algo_name, variant in orig_permutation[iter].items():
                    if variant == selected_variant:
                        algo = algo_name

                all_selected.append({
                    "movie_idx": selected_movie_idx,
                    "variant": selected_variant,
                    "algorithm": algo,
                    "mo": algo != "BETA",
                    "session": iter + 1
                })

        users_partial_lists = np.full((1, len(all_selected)), -1, dtype=np.int32)


        for i in range(N_ITERATIONS):
            already_seen = df_impressions_user[df_impressions_user.iteration < i + 1].movieId.values.astype(int).tolist()
            filter_out_movies = elicitation_selected + already_seen
            selected_movies = elicitation_selected + sum(d["selected"][:i], [])
            print(f"Iteration = {i}, num selected = {selected_movies}")
            movie_titles, scores, model = predict_ratings(selected_movies, filter_out_movies, return_model = True)
            #real = df_impressions_beta_user[df_impressions_beta_user.iteration == i + 1].movieId.values.astype(int).tolist()
            
            _, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies, k=len(sum(d["selected"], [])))
            wrapper.init()
            
            selected_so_far = len(sum(d["selected"][:i], []))
            selected_in_iteration = len(d["selected"][i])
            for idx in range(selected_so_far, selected_so_far + selected_in_iteration):
                
                selected_info = all_selected[idx]

                # Calculate support values
                supports = get_supports(users_partial_lists, wrapper.items,
                                                wrapper.extended_rating_matrix, wrapper.distance_matrix,
                                                wrapper.users_viewed_item, k=idx+1, n_users=wrapper.n_users)
                
                supports[0, :, :] = wrapper.normalizations[0](supports[0].T).T * wrapper.discount_sequences[0][idx]
                supports[1, :, :] = wrapper.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[1][idx]
                supports[2, :, :] = wrapper.normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[2][idx]
                ### End of support calculation ###
                

                new_row = [
                    participation,
                    selected_info["movie_idx"],
                    supports[0, 0, idx],
                    supports[1, 0, idx],
                    supports[2, 0, idx],
                    selected_info["session"],
                    idx,
                    selected_info["algorithm"],
                    selected_info["mo"]
                ]
                df_supports = pd.concat([df_supports, pd.DataFrame([new_row], columns=df_supports.columns)], axis=0)

                assert users_partial_lists[0, idx] == -1, f"{users_partial_lists}, idx={idx}"
                users_partial_lists[0, idx] = selected_info["movie_idx"]


            #print(f"@@ i={i}, real={real}, participation={participation}")
            #print(f"@@ predicted={x}, already_seen={already_seen}, elicitation_selected={elicitation_selected}, rest={sum(d['selected'][:i], [])}")
            
            
            #print(f"For iteration = {i + 1} we assume following recommendation: {x}, real was: {real}")
            #assert x == real
            # 
        assert not np.any(users_partial_lists[0] == -1), f"{users_partial_lists}"
        print(f"Participant = {participation} DONE after {time.perf_counter() - start_time}")
    return df_supports

def do_mmr(impression_indices, impression_scores, lmbda):
    partial_list = []
    candidate_items = set(impression_indices)

    while len(partial_list) < K:

        top_candidate = None
        top_candidate_score = None

        for movie_index, movie_score in zip(impression_indices, impression_scores):
            if movie_index not in candidate_items:
                continue

            # Maximal similarity between considered movie_index and already selected movies in the partial list
            if len(partial_list) == 0:
                max_sim = 0
            else:
                max_sim = loader.similarity_matrix[movie_index, partial_list].max()
            mmr = lmbda * movie_score - (1 - lmbda) * max_sim

            if top_candidate_score is None or mmr > top_candidate_score:
                top_candidate_score = mmr
                top_candidate = movie_index

        
        candidate_items.remove(top_candidate)
        partial_list.append(top_candidate)

    return partial_list

# Calculates p_g' as in paper
@functools.cache
def calc_p_g_1(genre, all_user_impressions):
    x1 = 0
    x2 = 0
    for single_usr_impressions in all_user_impressions:
        x1 += len([x for x in single_usr_impressions if genre in get_item_genres(x)])
        x2 += len(single_usr_impressions)

    p_g_1 = x1 / x2 #sum([num_items_having_the_genre[user] for user in users]) / sum([num_items_seen_by_user[user] for user in users])
    return p_g_1

# Probability of a genre
def p_g(genre, impression_indices, p_g_1, alfa = 0.5):
    
    num_items_seen_by_user = len(impression_indices)

    # Number of items that have the given genre
    # only items the user had interacted with are considered
    num_items_having_the_genre = len([x for x in impression_indices if genre in get_item_genres(x)])

    #p_g"
    p_g_2 = num_items_having_the_genre / num_items_seen_by_user # [0, 1]
    
    #p_g' -> passed as parameter, can be precalculated
    #p_g_1 = calc_p_g_1(genre, all_user_impressions)

    return (1 - alfa) * p_g_1[genre] + alfa * p_g_2

#### Coverage
# Calculate for each genre separately
# N is length of the recommendation list
# k is number of successes, that is, number of items belonging to that genre
# For each genre, recommendation list is sequence of bernouli trials, and each item in the list having the genre is considered to be a success
# Calculate probability of k successes in N trials
def binomial_probability(N, k, genre, impression_indices, p_g_1):
    p = p_g(genre, impression_indices, p_g_1)
    return scipy.special.comb(N, k) * np.power(p, k) * np.power(1.0 - p, N - k) 

# Return all genres that given movie has
def get_item_genres(item_index):
    item_id = loader.movie_index_to_id[item_index]
    genres = loader.movies_df_indexed.loc[item_id].genres.split("|")
    return genres

# Return all genres that all movies in the recommendation list have altogether (union over their genres)
def get_list_genres(rec_list):
    all_genres = set()
    for item in rec_list:
        all_genres = all_genres.union(get_item_genres(item))
    return all_genres

# Return all the genres available in the dataaset
@functools.cache
def get_all_genres():
    all_genres = set()
    for row in loader.movies_df_indexed.genres.unique():
        genres = row.split("|")
        all_genres = all_genres.union(genres)
    return all_genres

# Coverage as in the Binomial algorithm paper
def coverage(impression_indices, p_g_1, rec_list):
    all_genres = get_all_genres()
    rec_list_genres = get_list_genres(rec_list)
    not_rec_list_genres = all_genres - rec_list_genres

    N = len(rec_list)
    
    prod = 1
    for g in not_rec_list_genres:
        prod *= np.power(binomial_probability(N, 0, g, impression_indices, p_g_1), 1.0 / len(all_genres))

    return prod

# Genre redundancy as in the Binomial algorithm paper
def genre_redundancy(impression_indices, p_g_1, g, k, N):
    s = 0
    for l in range(1, k):
        # We want P(x_g = l | X_g > 0) so rewrite it as P(x_g = l & X_g > 0) / P(X_g > 0)
        # P(x_g = l & X_g > 0) happens when P(x_g = l) is it already imply X_g > 0
        # so we further simplify this as P(x_g = l) / P(X_g > 0) and P(X_g > 0) can be set to 1 - P(X_g = 0)
        # so we end up with
        # P(x_g = l) / (1 - P(X_g = 0))
        s += (binomial_probability(N, l, g, impression_indices, p_g_1) / (1.0 - binomial_probability(N, 0, g, impression_indices, p_g_1)))

    return 1.0 - s

# Return number of movies with a given genre g (only check movies from recommendation list rec_list)
def get_num_movies_with_genre(rec_list, g):
    k = 0
    for movie_idx in rec_list:
        movie_genres = get_item_genres(movie_idx)
        if g in movie_genres:
            k += 1
    return k

# NonRed as in the Binomial paper
def non_red(impression_indices, p_g_1, rec_list):
    #all_genres = get_all_genres()
    rec_list_genres = get_list_genres(rec_list)

    N = len(rec_list)
    prod = 1.0
    for g in rec_list_genres:
        num_movies_with_genre = get_num_movies_with_genre(rec_list, g)
        prod *= np.power(genre_redundancy(impression_indices, p_g_1, g, num_movies_with_genre, N), 1.0 / len(rec_list_genres))

    return prod

# Binomial diversity reranking scoring function
def binomial_diversity(impression_indices, p_g_1, rec_list):
    return coverage(impression_indices, p_g_1, rec_list) * non_red(impression_indices, p_g_1, rec_list)

# Perform binomial reranking
def do_binomial_reranking(impression_indices, impression_scores, p_g_1, scores_mean, scores_std, diversity_mean, diversity_std, lmbda):
    partial_list = []
    candidate_items = set(impression_indices)

    bin_diversities = []

    while len(partial_list) < K:

        top_candidate = None
        top_candidate_score = None

        binomial_diversity_cached = binomial_diversity(impression_indices, p_g_1, partial_list)

        for movie_index, movie_score in zip(impression_indices, impression_scores):
            if movie_index not in candidate_items:
                continue

            bin_diversity = binomial_diversity(impression_indices, p_g_1, partial_list + [movie_index]) - binomial_diversity_cached
            

            bin_diversities.append(bin_diversity)

            if scores_mean is None:
                movie_score_normed = movie_score
                bin_diversity_normed = bin_diversity
            else:
                movie_score_normed = (movie_score - scores_mean) / scores_std
                bin_diversity_normed = (bin_diversity - diversity_mean) / diversity_std

            score = (1 - lmbda) * movie_score_normed + lmbda * bin_diversity_normed

            if top_candidate_score is None or score > top_candidate_score:
                top_candidate_score = score
                top_candidate = movie_index

        
        candidate_items.remove(top_candidate)
        partial_list.append(top_candidate)

    return partial_list, bin_diversities

def do_support_reranking(impression_indices, wrapper, lmbda):
    partial_list = []
    candidate_items = set(impression_indices)

    while len(partial_list) < K:

        top_candidate = None
        top_candidate_score = None

        ### Support calculation ###
        users_partial_lists = np.full((wrapper.extended_rating_matrix.shape[0], len(partial_list) + 1), -1, dtype=np.int32)
        users_partial_lists[:, :len(partial_list)] = partial_list
        
        # Calculate support values
        supports = get_supports(users_partial_lists, wrapper.items,
                                        wrapper.extended_rating_matrix, wrapper.distance_matrix,
                                        wrapper.users_viewed_item, k=len(partial_list)+1, n_users=wrapper.n_users)
        
        supports[0, :, :] = wrapper.normalizations[0](supports[0].T).T * wrapper.discount_sequences[0][len(partial_list)]
        supports[1, :, :] = wrapper.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[1][len(partial_list)]
        ### End of support calculation ###

        for movie_index in impression_indices:
            if movie_index not in candidate_items:
                continue

            movie_supports = {
                "relevance": supports[0, 0, movie_index],
                "diversity": supports[1, 0, movie_index]
            }

            score = lmbda * movie_supports["relevance"] + (1 - lmbda) * movie_supports["diversity"]

            if top_candidate_score is None or score > top_candidate_score:
                top_candidate_score = score
                top_candidate = movie_index
        
        candidate_items.remove(top_candidate)
        partial_list.append(top_candidate)

    return partial_list


def precalculate_normalizations(df_interaction, df_elicitation_selections, df_impressions, lmbda, out_path):
    all_user_impressions = []
    for participation in df_interaction.participation.unique():
        df_impressions_user = df_impressions[df_impressions.userId == participation]
        all_impressions = df_impressions_user.movieId.values.astype(int).tolist()
        all_user_impressions.append(all_impressions)

    print(f"Precalculating all p_g_1 for each genre out of: {len(get_all_genres())}")
    p_g_1 = {} # for each genre
    for g in get_all_genres():
        x1 = 0.0
        x2 = 0.0
        for user_impressions in all_user_impressions:
            x1 += len([x for x in user_impressions if g in get_item_genres(x)])
            x2 += len(user_impressions)
        p_g_1[g] = x1 / x2

    print(f"Built impressions, len = {len(all_user_impressions)}, shp of participations = {df_interaction.participation.unique().shape}")

    assert df_interaction.shape[0] == df_interaction.participation.unique().shape[0], f"{df_interaction.shape} != {df_interaction.participation.unique().shape}"
    n_participations = 0

    all_bin_diversities = []

    for _, row in df_interaction.iterrows():
        participation = row.participation

        d = json.loads(row.data)
        
        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()
        df_impressions_user = df_impressions[df_impressions.userId == participation]
        
        all_impressions = set(df_impressions_user.movieId.values.astype(int).tolist())

        selected_movies = elicitation_selected
        filter_out_movies = elicitation_selected
        movie_titles, scores, model = predict_ratings(selected_movies, filter_out_movies, return_model = True)


        scores_mean = scores.mean()
        scores_std = scores.std()


        movie_titles = movie_titles[:K*5]
        scores = scores[:K*5]

        gold_selection_indices = sum(d["selected"], [])
        #assert len(gold_selection_indices) > 0, f"Participant {row} did not select anything"

        movie_indices = [movie_title_to_idx[t] for t in movie_titles]

        impression_indices = []
        impression_scores = []

        for movie_index, movie_title, movie_score in zip(movie_indices, movie_titles, scores):
            if movie_index in all_impressions:
                impression_indices.append(movie_index)
                impression_scores.append(movie_score)
        
        start_time = time.perf_counter()
        binomial_list, bin_diversities = do_binomial_reranking(impression_indices, impression_scores, p_g_1, None, None, None, None, lmbda = lmbda)
        n_participations += 1
        print(f"Took: {time.perf_counter() - start_time}, n_participations = {n_participations + 1}")

        all_bin_diversities.extend(bin_diversities)

    print(f"Done with calculating diversities")
    print(bin_diversities)


    with open(out_path, "wb") as f:
        arr = np.array(all_bin_diversities)
        pickle.dump({
            "all_bin_diversities": all_bin_diversities,
            "diversity_mean": arr.mean(),
            "diversity_std": arr.std()
        }, f)
        print(f"Diversity mean: {arr.mean()}, std: {arr.std()}")


def reranking(df_interaction, df_elicitation_selections, df_impressions, lmbda, diversity_normalization_path, out_path):
    print(f"Doing reranking")

    with open(diversity_normalization_path, "rb") as f:
        x = pickle.load(f)
        div_mean = x["diversity_mean"]
        div_std = x["diversity_std"]
    
    all_user_impressions = []
    for participation in df_interaction.participation.unique():
        df_impressions_user = df_impressions[df_impressions.userId == participation]
        all_impressions = df_impressions_user.movieId.values.astype(int).tolist()
        all_user_impressions.append(all_impressions)


    print(f"Precalculating all p_g_1 for each genre out of: {len(get_all_genres())}")
    p_g_1 = {} # for each genre
    for g in get_all_genres():
        x1 = 0.0
        x2 = 0.0
        for user_impressions in all_user_impressions:
            x1 += len([x for x in user_impressions if g in get_item_genres(x)])
            x2 += len(user_impressions)
        p_g_1[g] = x1 / x2

    print(f"Built impressions, len = {len(all_user_impressions)}, shp of participations = {df_interaction.participation.unique().shape}")

    algorithms = {"baseline", "mmr", "supports", "binomial"}
    metrics = {"diversity", "recall", "coverage", "selection_diversity", "selection_coverage"}
    per_algo_results = { algo: { metric: [] for metric in metrics } for algo in algorithms }

    assert df_interaction.shape[0] == df_interaction.participation.unique().shape[0], f"{df_interaction.shape} != {df_interaction.participation.unique().shape}"
    n_participations = 0
    
    full_reranking_results = {
        "participations": [],
        "lists": {}
    }


    for _, row in df_interaction.iterrows():
        participation = row.participation
        full_reranking_results["participations"].append(participation)
        full_reranking_results["lists"][participation] = {}

        d = json.loads(row.data)
        
        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()
        df_impressions_user = df_impressions[df_impressions.userId == participation]
        
        all_impressions = set(df_impressions_user.movieId.values.astype(int).tolist())

        selected_movies = elicitation_selected
        filter_out_movies = elicitation_selected
        movie_titles, scores, model = predict_ratings(selected_movies, filter_out_movies, return_model = True)


        scores_mean = scores.mean()
        scores_std = scores.std()


        movie_titles = movie_titles[:K*5]
        scores = scores[:K*5]

        gold_selection_indices = sum(d["selected"], [])
        #assert len(gold_selection_indices) > 0, f"Participant {row} did not select anything"

        movie_indices = [movie_title_to_idx[t] for t in movie_titles]

        impression_indices = []
        impression_scores = []

        for movie_index, movie_title, movie_score in zip(movie_indices, movie_titles, scores):
            if movie_index in all_impressions:
                impression_indices.append(movie_index)
                impression_scores.append(movie_score)
        
        print(f"Relevance based list = {movie_indices[:K]}")
        per_algo_results["baseline"]["diversity"].append(intra_list_diversity(movie_indices[:K]))
        per_algo_results["baseline"]["recall"].append(recall(movie_indices[:K], gold_selection_indices))
        per_algo_results["baseline"]["coverage"].append(topic_coverage(movie_indices[:K], impression_indices))
        per_algo_results["baseline"]["selection_diversity"].append(intra_list_diversity_on_hits(movie_indices[:K], gold_selection_indices))
        per_algo_results["baseline"]["selection_coverage"].append(topic_coverage_on_hits(movie_indices[:K], gold_selection_indices))
        #print(f"\tdiversity = {per_algo_results['baseline']['diversity'][-1]}, recall={per_algo_results['baseline']['recall'][-1]}, topic coverage={per_algo_results['baseline']['coverage'][-1]}")
        for m, lst in per_algo_results["baseline"].items():
            print(f"\t{m}={lst[-1]}")
        full_reranking_results["lists"][participation]["baseline"] = {
            "list": movie_indices[:K],
            "diversity": per_algo_results["baseline"]["diversity"][-1],
            "recall": per_algo_results["baseline"]["recall"][-1],
            "coverage": per_algo_results["baseline"]["coverage"][-1],
            "selection_diversity": per_algo_results["baseline"]["selection_diversity"][-1],
            "selection_coverage": per_algo_results["baseline"]["selection_coverage"][-1]
        }
        

        mmr_list = do_mmr(impression_indices, impression_scores, lmbda = lmbda)
        print(f"MMR list = {mmr_list}")
        per_algo_results["mmr"]["diversity"].append(intra_list_diversity(mmr_list))
        per_algo_results["mmr"]["recall"].append(recall(mmr_list, gold_selection_indices))
        per_algo_results["mmr"]["coverage"].append(topic_coverage(mmr_list, impression_indices))
        per_algo_results["mmr"]["selection_diversity"].append(intra_list_diversity_on_hits(mmr_list, gold_selection_indices))
        per_algo_results["mmr"]["selection_coverage"].append(topic_coverage_on_hits(mmr_list, gold_selection_indices))
        #print(f"\tdiversity = {per_algo_results['mmr']['diversity'][-1]}, recall={per_algo_results['mmr']['recall'][-1]}, topic coverage={per_algo_results['mmr']['coverage'][-1]}")
        for m, lst in per_algo_results["mmr"].items():
            print(f"\t{m}={lst[-1]}")
        full_reranking_results["lists"][participation]["mmr"] = {
            "list": movie_indices[:K],
            "diversity": per_algo_results["mmr"]["diversity"][-1],
            "recall": per_algo_results["mmr"]["recall"][-1],
            "coverage": per_algo_results["mmr"]["coverage"][-1],
            "selection_diversity": per_algo_results["mmr"]["selection_diversity"][-1],
            "selection_coverage": per_algo_results["mmr"]["selection_coverage"][-1]
        }

        _, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies)
        wrapper.init()
        supports_list = do_support_reranking(impression_indices, wrapper, lmbda = lmbda)
        print(f"SUPPORTS list = {supports_list}")
        per_algo_results["supports"]["diversity"].append(intra_list_diversity(supports_list))
        per_algo_results["supports"]["recall"].append(recall(supports_list, gold_selection_indices))
        per_algo_results["supports"]["coverage"].append(topic_coverage(supports_list, impression_indices))
        per_algo_results["supports"]["selection_diversity"].append(intra_list_diversity_on_hits(supports_list, gold_selection_indices))
        per_algo_results["supports"]["selection_coverage"].append(topic_coverage_on_hits(supports_list, gold_selection_indices))
        #print(f"\tdiversity = {per_algo_results['supports']['diversity'][-1]}, recall={per_algo_results['supports']['recall'][-1]}, topic coverage={per_algo_results['supports']['coverage'][-1]}")
        for m, lst in per_algo_results["supports"].items():
            print(f"\t{m}={lst[-1]}")
        full_reranking_results["lists"][participation]["supports"] = {
            "list": movie_indices[:K],
            "diversity": per_algo_results["supports"]["diversity"][-1],
            "recall": per_algo_results["supports"]["recall"][-1],
            "coverage": per_algo_results["supports"]["coverage"][-1],
            "selection_diversity": per_algo_results["supports"]["selection_diversity"][-1],
            "selection_coverage": per_algo_results["supports"]["selection_coverage"][-1]
        }

        start_time = time.perf_counter()
        binomial_list, _ = do_binomial_reranking(impression_indices, impression_scores, p_g_1, scores_mean, scores_std, div_mean, div_std, lmbda = lmbda)
        print(f"BINOMIAL list = {binomial_list}")
        per_algo_results["binomial"]["diversity"].append(intra_list_diversity(binomial_list))
        per_algo_results["binomial"]["recall"].append(recall(binomial_list, gold_selection_indices))
        per_algo_results["binomial"]["coverage"].append(topic_coverage(binomial_list, impression_indices))
        per_algo_results["binomial"]["selection_diversity"].append(intra_list_diversity_on_hits(binomial_list, gold_selection_indices))
        per_algo_results["binomial"]["selection_coverage"].append(topic_coverage_on_hits(binomial_list, gold_selection_indices))
        #print(f"\tdiversity = {per_algo_results['binomial']['diversity'][-1]}, recall={per_algo_results['binomial']['recall'][-1]}, topic coverage={per_algo_results['binomial']['coverage'][-1]}")
        for m, lst in per_algo_results["binomial"].items():
            print(f"\t{m}={lst[-1]}")
        full_reranking_results["lists"][participation]["binomial"] = {
            "list": movie_indices[:K],
            "diversity": per_algo_results["binomial"]["diversity"][-1],
            "recall": per_algo_results["binomial"]["recall"][-1],
            "coverage": per_algo_results["binomial"]["coverage"][-1],
            "selection_diversity": per_algo_results["binomial"]["selection_diversity"][-1],
            "selection_coverage": per_algo_results["binomial"]["selection_coverage"][-1]
        }

        print(f"Took: {time.perf_counter() - start_time}")




        n_participations += 1

        if n_participations % 20 == 0:
            print(f"Average results after 10 iterations are:")

            for algo, results in per_algo_results.items():
                for m, values in results.items():
                    print(f"algo={algo},metric={m},mean={sum(values)/len(values)}")
    

    print("Done, very final results are:")
    with open(out_path, "wb") as f:
        full_reranking_results["objectives"] = per_algo_results
        full_reranking_results["mean_results"] = {}

        for algo, results in per_algo_results.items():
            full_reranking_results["mean_results"][algo] = {}
            for m, values in results.items():
                print(f"algo={algo},metric={m},mean={sum(values)/len(values)}")
                full_reranking_results["mean_results"][algo][m] = sum(values)/len(values)

        pickle.dump(full_reranking_results, f)
        

    

def intra_list_diversity(rec_list):
    assert len(rec_list) > 0
    distance_matrix = 1.0 - loader.similarity_matrix
    n = len(rec_list)
    div = 0
    for i in range(n):
        for j in range(i):
            div += distance_matrix[rec_list[i], rec_list[j]]
    return div / ((n - 1) * n / 2)

def intra_list_diversity_on_hits(rec_list, all_user_selections):
    distance_matrix = 1.0 - loader.similarity_matrix

    rec_list_hits = [x for x in rec_list if x in all_user_selections]
    
    n = len(rec_list_hits)

    if n == 0:
        return 0.0
    elif n == 1:
        return distance_matrix[rec_list_hits[0]].sum() / (distance_matrix.shape[0] - 1)

    div = 0
    for i in range(n):
        for j in range(i):
            div += distance_matrix[rec_list_hits[i], rec_list_hits[j]]
    return div / ((n - 1) * n / 2)

# Recall@K
def recall(rec_list, all_user_selections):
    rec_list_correct = [x for x in rec_list if x in all_user_selections]
    if min(len(all_user_selections), len(rec_list)) == 0:
        return 0.0
    return len(rec_list_correct) / min(len(all_user_selections), len(rec_list))
                
def topic_coverage(rec_list, all_user_impressions):
    #all_user_seen_genres = get_list_genres(all_user_impressions)
    rec_list_genres = get_list_genres(rec_list)

    return len(rec_list_genres) / len(get_all_genres()) #len(all_user_seen_genres)

def topic_coverage_on_hits(rec_list, all_user_selections):
    rec_list_hits = [x for x in rec_list if x in all_user_selections]
    rec_list_genres = get_list_genres(rec_list_hits)
    return len(rec_list_genres) / len(get_all_genres())


def calc_list_supports(real_list, wrapper):
    users_partial_lists = np.full((1, len(real_list)), -1, dtype=np.int32)
    final_supports = []
    for idx, m_index in enumerate(real_list):
        
        # Calculate support values
        supports = get_supports(users_partial_lists, wrapper.items,
                                        wrapper.extended_rating_matrix, wrapper.distance_matrix,
                                        wrapper.users_viewed_item, k=idx+1, n_users=wrapper.n_users)
        
        supports[0, :, :] = wrapper.normalizations[0](supports[0].T).T * wrapper.discount_sequences[0][idx]
        supports[1, :, :] = wrapper.normalizations[1](supports[1].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[1][idx]
        supports[2, :, :] = wrapper.normalizations[2](supports[2].reshape(-1, 1)).reshape((supports.shape[1], -1)) * wrapper.discount_sequences[2][idx]
        
        ### End of support calculation ###
        final_supports.append({
            "movie_idx": m_index,
            "support": {
                "relevance": supports[0, 0, idx],
                "diversity": supports[1, 0, idx],
                "novelty": supports[2, 0, idx]
            },
            "rank": idx
        })
        users_partial_lists[0, idx] = m_index
    return final_supports

def show_differences(df_interaction, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx, df_selections):

    results = pd.DataFrame(columns=["userId", "session", "FT_vs_NO_FT", "FT_vs_BETA_FT", "FT_vs_REAL"])
    results2 = pd.DataFrame(columns=["userId", "session", "rank", "movieId", "kind", "relevanceGain", "diversityGain", "noveltyGain"])

    for _, row in df_interaction.iterrows():
        d = json.loads(row.data)
        participation = row.participation

        elicitation_selected = df_elicitation_selections[df_elicitation_selections.userId == participation].movieId.values.astype(int).tolist()

        df_impressions_user = df_impressions[df_impressions.userId == participation]
        df_impressions_beta_user = df_impressions_beta[df_impressions_beta.userId == participation]
        df_selections_user = df_selections[df_selections.userId == participation]
        print(df_selections_user)

        for i in range(N_ITERATIONS):
            already_seen = df_impressions_user[df_impressions_user.iteration < i + 1].movieId.values.astype(int).tolist()
            filter_out_movies = elicitation_selected + already_seen
            selected_movies = elicitation_selected + sum(d["selected"][:i], [])

            #print(f"i={i}, D selected = {d['selected']}, D len = {len(d['selected'])}, user selections from CSV = {df_selections_user[(df_selections_user.session < i + 1)].movieId.values.astype(int).tolist()}")

            real = df_impressions_beta_user[df_impressions_beta_user.iteration == i + 1].movieId.values.astype(int).tolist()

            alternative_source_selections = df_selections_user[(df_selections_user.session < i + 1)].movieId.values.astype(int).tolist()
            alternative_source_selections = list(set(sum(d["selected"][:i], [])).intersection(alternative_source_selections))
            #assert set(selected_movies) == set(elicitation_selected + alternative_source_selections), f"{selected_movies} != {elicitation_selected + alternative_source_selections}"

            beta_selections = df_selections_user[(df_selections_user.session < i + 1) & (df_selections_user.mo == False)].movieId.values.astype(int).tolist()
            print(f"Beta selections before: {beta_selections}")
            beta_selections = list(set(sum(d["selected"][:i], [])).intersection(beta_selections)) ## Getting rid of deselections
            print(f"Beta selections after: {beta_selections}")

            top_k_fine_tuned, model = recommend_2_3(selected_movies, filter_out_movies, return_model=True)
            top_k_fine_tuned = [int(x["movie_idx"]) for x in top_k_fine_tuned]
            _, wrapper = prepare_wrapper(selected_movies, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies)
            wrapper.init()
            res = calc_list_supports(top_k_fine_tuned, wrapper)
            #assert top_k_fine_tuned == [r["movie_idx"] for r in res], f"{top_k_fine_tuned} != {res}"
            for r in res:
                new_row = [participation, i + 1, r["rank"], r["movie_idx"], "ft", r["support"]["relevance"], r["support"]["diversity"], r["support"]["novelty"]]
                results2 = pd.concat([results2, pd.DataFrame([new_row], columns=results2.columns)])


            top_k_no_fine_tuned, model = recommend_2_3(elicitation_selected, filter_out_movies, return_model=True)
            top_k_no_fine_tuned = [int(x["movie_idx"]) for x in top_k_no_fine_tuned]
            _, wrapper = prepare_wrapper(elicitation_selected, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies)
            wrapper.init()
            res = calc_list_supports(top_k_no_fine_tuned, wrapper)
            #assert top_k_no_fine_tuned == [r["movie_idx"] for r in res], f"{top_k_no_fine_tuned} != {res}"
            for r in res:
                new_row = [participation, i + 1, r["rank"], r["movie_idx"], "no_ft", r["support"]["relevance"], r["support"]["diversity"], r["support"]["novelty"]]
                results2 = pd.concat([results2, pd.DataFrame([new_row], columns=results2.columns)])
            
            top_k_beta_fine_tuned, model = recommend_2_3(elicitation_selected + beta_selections, filter_out_movies, return_model=True)
            top_k_beta_fine_tuned = [int(x["movie_idx"]) for x in top_k_beta_fine_tuned]
            _, wrapper = prepare_wrapper(elicitation_selected + beta_selections, model, weighted_average_strategy, np.array([1.0, 0.0, 0.0]), filter_out_movies)
            wrapper.init()
            res = calc_list_supports(top_k_beta_fine_tuned, wrapper)
            #assert top_k_beta_fine_tuned == [r["movie_idx"] for r in res], f"{top_k_beta_fine_tuned} != {res}"
            for r in res:
                new_row = [participation, i + 1, r["rank"], r["movie_idx"], "beta_ft", r["support"]["relevance"], r["support"]["diversity"], r["support"]["novelty"]]
                results2 = pd.concat([results2, pd.DataFrame([new_row], columns=results2.columns)])
            
            ft_vs_no_ft = len(set(top_k_fine_tuned).intersection(top_k_no_fine_tuned))
            ft_vs_beta_ft = len(set(top_k_fine_tuned).intersection(top_k_beta_fine_tuned))
            ft_vs_real = len(set(top_k_fine_tuned).intersection(real))

            print(f"#### [{participation}, {i + 1}] Intersection sizes: FT vs NO_FT = {ft_vs_no_ft}, FT vs BETA_FT = {ft_vs_beta_ft}, @@ FT vs REAL_FT = {ft_vs_real} @@")
            
            new_row = [participation, i + 1, ft_vs_no_ft, ft_vs_beta_ft, ft_vs_real]
            results = pd.concat([results, pd.DataFrame([new_row], columns=results.columns)])

            # Simulate whole recommendation lists
            


            # df_ratings_user = pd.DataFrame({
            #     "userId": np.repeat(participation, scores.size),
            #     "movieId": movie_indices,
            #     "rawRating": scores,
            #     "session": np.repeat(i + 1, scores.size)    
            # })
            
            #print(df_ratings_user.head())
            #df_ratings = pd.concat([df_ratings, df_ratings_user], axis=0)
            #print("DF_ratings =")
            #print(df_ratings.head())

            #real = df_impressions_beta_user[df_impressions_beta_user.iteration == i + 1].movieId.values.astype(int).tolist()
            
            #print(f"@@ i={i}, real={real}, participation={participation}")
            #print(f"@@ predicted={x}, already_seen={already_seen}, elicitation_selected={elicitation_selected}, rest={sum(d['selected'][:i], [])}")
            
            
            #print(f"For iteration = {i + 1} we assume following recommendation: {x}, real was: {real}")
            #assert x == real
            # 

    #return df_ratings
    return results, results2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--reranking", action="store_true", default=False)
    parser.add_argument("--gen-ratings", action="store_true", default=False)
    parser.add_argument("--gen-beta-supports", action="store_true", default=False)
    parser.add_argument("--precalculate-normalizations", action="store_true", default=False)
    parser.add_argument("--lmbda", type=float)
    parser.add_argument("--participation-path", type=str, default="C:/Users/PD/Documents/MFF/beyinterecsys/data/18-4-2023 19-18/participation-export.json")
    parser.add_argument("--interaction-path", type=str, default="C:/Users/PD/Documents/MFF/beyinterecsys/data/18-4-2023 19-18/interaction-export.json")
    args = parser.parse_args()

    df_interaction = pd.read_json(args.interaction_path, encoding='utf-8')
    df_interaction_raw = df_interaction.copy()
    df_participation = pd.read_json(args.participation_path, encoding="utf-8")
    df_participation = df_participation[df_participation.user_study_id >= 4]
    df_participation = df_participation.set_index("id")
    df_participation = df_participation[df_participation.participant_email != "a@a.a"]
    df_participation  = df_participation[df_participation.time_finished.notna()]
    df_interaction = df_interaction[df_interaction.participation.isin(df_participation.index)]
    df_elicitation = df_interaction[df_interaction.interaction_type == "elicitation-ended"]
    
    df_interaction = df_interaction.apply(set_iteration, axis=1)
    df_interaction['iteration'] = df_interaction.groupby(['participation'], sort=False, group_keys=False)['iteration'].apply(lambda x: x.ffill())
    df_interaction = df_interaction.dropna()
    df_interaction.iteration = df_interaction.iteration.astype(int)
    
    df_interaction_full = df_interaction.copy()
    df_interaction = df_interaction[(df_interaction.interaction_type == "iteration-ended") & (df_interaction.iteration == N_ITERATIONS)]
    df_interaction = df_interaction.drop_duplicates(subset='participation', keep="last") # If someone pressed back button and finished iteration twice (happens rarely)
    
    
    df_elicitation_selections = create_elicitation_selections_csv(df_interaction_raw)
    
    df_impressions = create_impressions_csv(df_interaction_full)
    df_impressions_beta = create_beta_impressions_csv(df_interaction_full)
    df_selections = create_selections_csv(df_interaction_full)


    movie_title_to_idx = dict()
    for movie_id, row in loader.movies_df_indexed.iterrows():
        movie_title_to_idx[bytes(row.title, "UTF-8")] = loader.movie_id_to_index[movie_id]

    
    if args.gen_ratings:
        print(f"Generating ratings")
        calc_ratings(df_interaction, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx).to_csv("./ratings.csv", index=False)
    
    div_path = "diversity_normalizations.pkl"

    if args.precalculate_normalizations:
        print(f"Precalculating normalizations")
        precalculate_normalizations(df_interaction, df_elicitation_selections, df_impressions, args.lmbda, out_path=div_path)

    if args.reranking:
        print(f"Starting re-ranking with lambda={args.lmbda}")
        reranking(df_interaction, df_elicitation_selections, df_impressions, args.lmbda, div_path, out_path=f"reranking_{args.lmbda}.pkl")

    if args.gen_beta_supports:
        print(f"Generating beta supports")
        df_weights = create_weights_csv(df_interaction_full)
        calc_beta_supports(df_interaction, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx, df_weights).to_csv("./beta_supports.csv", index=False)

    #calc_adjusted_supports(df_interaction, df_elicitation, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx).to_csv("./tmp_1.csv", index=False)
    #calc_adjusted_supports_2(df_interaction, df_elicitation, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx).to_csv("./tmp_2.csv", index=False)

    #loader.movies_df.to_csv("movies_df_backup.csv", index=False)
    #np.save("./similarity_matrix_backup.npy", loader.similarity_matrix)
    #np.save("./rating_matrix_backup.npy", loader.rating_matrix)
    

    r1, r2 = show_differences(df_interaction, df_elicitation_selections, df_impressions, df_impressions_beta, movie_title_to_idx, df_selections)
    r1.to_csv("./ft_no_ft_comparison.csv", index=False)
    r2.to_csv("./ft_no_ft_recommendations.csv", index=False)