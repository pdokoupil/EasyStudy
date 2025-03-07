import itertools
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

import plugins.grs2024.settings.config_movie_lens as cfg

class AggregationStrategy(ABC):

    @staticmethod
    def getAggregator(strategy):
        if strategy == "ADD":
            return AdditiveAggregator()
        elif strategy == "AVG":
            return AverageAggregator()
        elif strategy == "LMS":
            return LeastMiseryAggregator()
        elif strategy == "BASE":
            return BaselinesAggregator()
        elif strategy == "GFAR":
            return GFARAggregator()
        elif strategy == "EPFuzzDA":
            return EPFuzzDAAggregator()
        elif strategy == "FAI":
            return FAIAggregator()
        elif strategy == "BDC":
            return BordaCountAggregator()
        elif strategy == "AVGNM":
            return AVGNoMiseryAggregator()
        elif strategy == "GreedyLM":
            return GreedyLMAggregator()
        elif strategy == "GreedyEXACT":
            return GreedyEXACTAggregator()
        elif strategy == "RLProp":
            return RLPropAggregator()
        elif strategy == "Random":
            return RandomAggregator()
        return None

    @abstractmethod
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        pass


class AdditiveAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        aggregated_df = group_ratings.groupby('item').sum()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['item', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['item'])
        return {"ADD": recommendation_list}

class AverageAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        aggregated_df = group_ratings.groupby('item').mean()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['item', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['item'])
        return {"AVG": recommendation_list}

class LeastMiseryAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        # aggregate using least misery strategy
        aggregated_df = group_ratings.groupby('item').min()
        aggregated_df = aggregated_df.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['item', 'predicted_rating']]
        recommendation_list = list(aggregated_df.head(recommendations_number)['item'])
        return {"LMS": recommendation_list}

class RandomAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        all_items = group_ratings.item.unique()
        return {"Random": np.random.choice(all_items, recommendations_number, replace=False)}

class BaselinesAggregator(AggregationStrategy):
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        # aggregate using least misery strategy
        aggregated_df = group_ratings.groupby('item').agg({"predicted_rating": [np.sum, np.prod, np.min, np.max]})
        aggregated_df = aggregated_df["predicted_rating"].reset_index()
        # additive

        add_df = aggregated_df.sort_values(by="sum", ascending=False).reset_index()[['item', 'sum']]
        add_recommendation_list = list(add_df.head(recommendations_number)['item'])
        # multiplicative
        mul_df = aggregated_df.sort_values(by="prod", ascending=False).reset_index()[['item', 'prod']]
        mul_recommendation_list = list(mul_df.head(recommendations_number)['item'])
        # least misery
        lms_df = aggregated_df.sort_values(by="amin", ascending=False).reset_index()[['item', 'amin']]
        lms_recommendation_list = list(lms_df.head(recommendations_number)['item'])
        # most pleasure
        mpl_df = aggregated_df.sort_values(by="amax", ascending=False).reset_index()[['item', 'amax']]
        mpl_recommendation_list = list(mpl_df.head(recommendations_number)['item'])
        return {
            "ADD": add_recommendation_list,
            "MUL": mul_recommendation_list,
            "LMS": lms_recommendation_list,
            "MPL": mpl_recommendation_list,
        }


class GFARAggregator(AggregationStrategy):
    # implements GFAR aggregation algorithm. For more details visit https://dl.acm.org/doi/10.1145/3383313.3412232

    # create an index-wise top-k selection w.r.t. list of scores
    def select_top_n_idx(self, score_df, top_n, top='max', sort=True):
        if top != 'max' and top != 'min':
            raise ValueError('top must be either Max or Min')
        if top == 'max':
            score_df.loc[:, "predicted_rating_rev"] = -score_df.loc[:, "predicted_rating"]

        select_top_n = min(top_n, len(score_df) - 1)
        top_n_ind = np.argpartition(score_df.predicted_rating_rev, select_top_n)[:select_top_n]
        top_n_df = score_df.iloc[top_n_ind]

        if sort:
            return top_n_df.sort_values("predicted_rating_rev")

        return top_n_df

    # borda count that is limited only to top-max_rel_items, if you are not in the top-max_rel_items, you get 0
    def get_borda_rel(self, candidate_group_items_df, max_rel_items):
        from scipy.stats import rankdata
        top_records = self.select_top_n_idx(candidate_group_items_df, max_rel_items, top='max', sort=False)

        rel_borda = rankdata(-top_records["predicted_rating_rev"].values, method='max') - 1
        # candidate_group_items_df.loc[top_records.index,"borda_score"] = rel_borda
        return (top_records.index, rel_borda)

    # runs GFAR algorithm for one group
    def gfar_algorithm(self, group_ratings, top_n: int, relevant_max_items: int, n_candidates: int):

        group_members = group_ratings.user.unique()
        group_size = len(group_members)

        localDF = group_ratings.copy()
        localDF["predicted_rating_rev"] = 0.0
        localDF["borda_score"] = 0.0
        localDF["p_relevant"] = 0.0
        localDF["prob_selected_not_relevant"] = 1.0
        localDF["marginal_gain"] = 0.0

        # filter-out completely irrelevant items to decrease computational complexity
        # top_candidates_ids_per_member = []
        # for uid in  group_members:
        #    per_user_ratings = group_ratings.loc[group_ratings.user == uid]
        #    top_candidates_ids_per_member.append(select_top_n_idx(per_user_ratings, n_candidates, sort=False)["item"].values)

        # top_candidates_idx = np.unique(np.array(top_candidates_ids_per_member))

        # get the candidate group items for each member
        # candidate_group_ratings = group_ratings.loc[group_ratings["items"].isin(top_candidates_idx)]

        for uid in group_members:
            per_user_candidates = localDF.loc[localDF.user == uid].copy(deep=False)
            borda_index, borda_score = self.get_borda_rel(per_user_candidates, relevant_max_items)
            localDF.loc[borda_index, "borda_score"] = borda_score

            total_relevance_for_user = localDF.loc[borda_index, "borda_score"].sum()
            localDF.loc[borda_index, "p_relevant"] = localDF.loc[borda_index, "borda_score"] / total_relevance_for_user

        selected_items = []

        # top-n times select one item to the final list
        for i in range(top_n):
            localDF.loc[:, "marginal_gain"] = localDF.p_relevant * localDF.prob_selected_not_relevant
            item_marginal_gain = localDF.groupby("item")["marginal_gain"].sum()
            # select the item with the highest marginal gain
            item_pos = item_marginal_gain.argmax()
            item_id = item_marginal_gain.index[item_pos]
            selected_items.append(item_id)

            # update the probability of selected items not being relevant
            for uid in group_members:
                winner_row = localDF.loc[((localDF["item"] == item_id) & (localDF["user"] == uid))]

                # only update if any record for user-item was found
                if winner_row.shape[0] > 0:
                    p_rel = winner_row["p_relevant"].values[0]
                    p_not_selected = winner_row["prob_selected_not_relevant"].values[0] * (1 - p_rel)

                    localDF.loc[localDF["user"] == uid, "prob_selected_not_relevant"] = p_not_selected

            # remove winning item from the list of candidates
            localDF.drop(localDF.loc[localDF["item"] == item_id].index, inplace=True)
        return selected_items

    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        selected_items = self.gfar_algorithm(group_ratings, recommendations_number, 20, 500)
        return {"GFAR": selected_items}


class EPFuzzDAAggregator(AggregationStrategy):
    # implements EP-FuzzDA aggregation algorithm. For more details visit https://dl.acm.org/doi/10.1145/3450614.3461679

    def ep_fuzzdhondt_algorithm(self, group_ratings, top_n, member_weights=None):
        group_members = group_ratings.user.unique()
        all_items = group_ratings["item"].unique()
        group_size = len(group_members)

        if not member_weights:
            member_weights = [1. / group_size] * group_size
        member_weights = pd.DataFrame(pd.Series(member_weights, index=group_members))

        localDF = group_ratings.copy()

        candidate_utility = pd.pivot_table(localDF, values="predicted_rating", index="item", columns="user",
                                           fill_value=0.0)
        candidate_sum_utility = pd.DataFrame(candidate_utility.sum(axis="columns"))

        total_user_utility_awarded = pd.Series(np.zeros(group_size), index=group_members)
        total_utility_awarded = 0.

        selected_items = []
        # top-n times select one item to the final list
        for i in range(top_n):


            prospected_total_utility = candidate_sum_utility + total_utility_awarded  # pd.DataFrame items x 1


            allowed_utility_for_users = pd.DataFrame(np.dot(prospected_total_utility.values, member_weights.T.values),
                                                     columns=member_weights.T.columns,
                                                     index=prospected_total_utility.index)


            # cap the item's utility by the already assigned utility per user
            unfulfilled_utility_for_users = allowed_utility_for_users.subtract(total_user_utility_awarded,
                                                                               axis="columns")
            unfulfilled_utility_for_users[unfulfilled_utility_for_users < 0] = 0
            
            candidate_user_relevance = pd.concat([unfulfilled_utility_for_users, candidate_utility]).groupby(level=0).min()
            
            candidate_relevance = candidate_user_relevance.sum(axis="columns")

            # remove already selected items
            candidate_relevance = candidate_relevance.loc[~candidate_relevance.index.isin(selected_items)]
            item_pos = candidate_relevance.argmax()
            item_id = candidate_relevance.index[item_pos]

            selected_items.append(item_id)

            winner_row = candidate_utility.loc[item_id, :]


            total_user_utility_awarded.loc[:] = total_user_utility_awarded.loc[:] + winner_row

            total_utility_awarded += winner_row.values.sum()

        return selected_items

    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        selected_items = self.ep_fuzzdhondt_algorithm(group_ratings, recommendations_number)
        return {"EPFuzzDA": selected_items}
    
class FAIAggregator(AggregationStrategy):
    # implements FAI aggregation algorithm
    def fai_algorithm(self, group_ratings, recommendations_number):
        selected_items = []        
        unique_users = group_ratings['user'].unique() # get all unique users in the group_ratings df
        
        for i in range(int(recommendations_number)):
            user_index = i % len(unique_users) # loop the number tracking the iterations (0, 1, ... len(unique_users), 0, 1, ...), so it doesnt try to access a user outside of the list of unique_users
            
            # print("user "+str(unique_users[user_index])+", looping index "+str(user_index)+" | linear index "+str(i))
            
            curr_user_ratings = group_ratings.loc[group_ratings['user'] == unique_users[user_index]] # only the ratings of current selected user index
            curr_user_ratings = curr_user_ratings.sort_values(by="predicted_rating", ascending=False).reset_index() # order the ratings so higher are on top
            curr_user_ratings = curr_user_ratings.loc[~curr_user_ratings['item'].isin(selected_items)] # remove all rows with item already on selected_items
            
            selected_item = curr_user_ratings['item'].iloc[0] # pick top row, with highest rating for current selected user index
            
            selected_items.append(selected_item) # append to final list
        
        return selected_items
    
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        selected_items = self.fai_algorithm(group_ratings, recommendations_number)
        return {"FAI": selected_items}

class BordaCountAggregator(AggregationStrategy):
    # implements borda count aggregation algorithm
    def bdc_algorithm(self, group_ratings, recommendations_number):
        from scipy.stats import rankdata
        
        unique_users = group_ratings['user'].unique() # get all unique users in the group_ratings df
        
        localDF = group_ratings.copy()
        localDF["borda_score"] = 0.0
        
        for uid in unique_users:
            per_user_candidates = localDF.loc[localDF.user == uid]
            
            borda_score = rankdata(per_user_candidates["predicted_rating"].values, method='min')
            borda_index = per_user_candidates.index
            
            localDF.loc[borda_index, "borda_score"] = borda_score
        
        aggregated_ratings = localDF.groupby('item').sum()
        aggregated_ratings = aggregated_ratings.sort_values(by="borda_score", ascending=False).reset_index()[
            ['item', 'predicted_rating', 'borda_score']]
        
        selected_items = list(aggregated_ratings.head(recommendations_number)['item'])
        
        return selected_items
    
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        selected_items = self.bdc_algorithm(group_ratings, recommendations_number)
        return {"BDC": selected_items}
    
class AVGNoMiseryAggregator(AggregationStrategy):
    # implements AVGNoMisery aggregation algorithm
    def avgnm_algorithm(self, group_ratings, recommendations_number, threshold=0):
        #print(f"Called AVGNoMisery with threshold={threshold}")
        #print(group_ratings)
        # should groupby items, .min() must be above threshold, this gives list of allowed items (can be empty)
        # then later check if item id is in this allowed list
        
        allowed_items = group_ratings.groupby('item', as_index=False).min() # Collect the worst ratings of items among all users
        allowed_items = allowed_items.loc[allowed_items['predicted_rating'] > threshold] # The worst rating within any user group (misery) must be above the threshold, else it's not allowed
        allowed_items = allowed_items['item'].tolist() # As a list to use by the filter later
        
        if len(allowed_items) == 0: # If there is no item allowed, just send back no list
            return []
        
        ordered_ratings = group_ratings.groupby('item').mean() # Get the list of items ordered by average (this is 'average no misery', so if it's within allowed_items, the best average wins)
        ordered_ratings = ordered_ratings.sort_values(by="predicted_rating", ascending=False).reset_index()[
            ['item', 'predicted_rating']]
        
        collected_items = ordered_ratings[ordered_ratings['item'].isin(allowed_items)] # Only collect those within the list of allowed items
        
        final_items = list(collected_items.head(recommendations_number)['item'])
        
        return final_items
    
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        selected_items = self.avgnm_algorithm(group_ratings, recommendations_number, 1) # thresh set at 2
        return {"AVGNM": selected_items}

import time
class GreedyLMAggregator(AggregationStrategy):
    def __init__(self):
        self.rel_max = None
        self.group_ratings = None

    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        self.group_ratings = group_ratings
        self.rel_max = group_ratings.predicted_rating.max()

        recommendation = np.empty([0], dtype=np.int32)
        items = group_ratings.item.unique()
        users = group_ratings.user.unique()
        lam = 0.5

        rating_matrix = group_ratings.pivot_table(columns="item", index="user", values="predicted_rating")
        item_to_item_index = {item: item_index for item, item_index in zip(rating_matrix.columns, range(len(rating_matrix.columns)))}
        #print("MAPPING", item_to_item_index)
        
        rating_matrix = rating_matrix.values
        assert rating_matrix.shape == (users.size, items.size)

        #user_top_ks_ratings = np.take_along_axis(rating_matrix, rating_matrix.argpartition(-recommendations_number)[:,-recommendations_number:], axis=-1)
        
        user_top_ks_indices = np.argsort(-rating_matrix)[:, :recommendations_number]
        user_top_ks_ratings = np.take_along_axis(rating_matrix, user_top_ks_indices, axis=-1)
        
        while len(recommendation) < recommendations_number:
            best_item = None
            for item in items:
                test_set = np.concatenate((recommendation, [item]))
                test_set_indices = np.array(list(map(item_to_item_index.get, test_set)))

                #individual_utilities = []
                
                #start_time = time.perf_counter()
                individual_utilities = rating_matrix[:, test_set_indices].sum(axis=1) / user_top_ks_ratings[:, :item.size].sum(axis=1)
                #print(f"Vectorized individual utilities took: {time.perf_counter() - start_time}, total_items = {len(items)}")
                
                #start_time = time.perf_counter()
                #for user in users:
                #    individual_utilities.append(self.individual_utility_proportional(user, test_set))
                #print(f"All individual utilities took: {time.perf_counter() - start_time}, total_items = {len(items)}")
                
                #print(individual_utilities)
                #print(individual_utilities_2.tolist())

                #assert False

                # sw = self.social_welfare(users, test_set)
                # f = self.fairness(users, test_set)
                sw = np.mean(individual_utilities)
                f = np.min(individual_utilities)

                score = lam * sw + (1 - lam) * f

                if best_item is None or best_item[1] < score:
                    best_item = (item, score)
                
            recommendation = np.append(recommendation, best_item[0])
            items = items[items != best_item[0]]

        return {"GreedyLM": recommendation}

    def rel(self, user, item):
        user_match = self.group_ratings.user == user
        item_match = self.group_ratings.item == item

        return self.group_ratings.loc[user_match & item_match, 'predicted_rating'].to_numpy()[0]

    def individual_utility_average(self, user, items):
        # acc = 0
        # for item in items:
        #     acc += self.rel(user, item)
        # return acc / (len(items) * self.rel_max)

        user_match = self.group_ratings.user == user
        total_sum = self.group_ratings.loc[user_match & self.group_ratings.item.isin(items)].predicted_rating.sum()
        return total_sum / (len(items) * self.rel_max)
    
    def individual_utility_proportional(self, user, items):
        k = len(items)
        user_match = self.group_ratings.user == user

        top_k_sum = self.group_ratings.loc[user_match].nlargest(k, 'predicted_rating').predicted_rating.sum()
        total_sum = self.group_ratings.loc[user_match & self.group_ratings.item.isin(items)].predicted_rating.sum()
        return total_sum / top_k_sum

    def social_welfare(self, group, items):
        acc = 0
        for user in group:
            acc += self.individual_utility_average(user, items)

        return acc / len(group)

    def fairness(self, group, items):
        cur_min = None
        for user in group:
            ind_util = self.individual_utility_average(user, items)
            if cur_min is None or cur_min > ind_util:
                cur_min = ind_util

        return cur_min


    
class RLPropAggregator(AggregationStrategy):

    NEG_INF = int(-10e6)

    def __init__(self):
        self.with_norm = False

    def mask_scores(self, scores, seen_items_mask):
        # Ensure seen items get lowest score of 0
        # Just multiplying by zero does not work when scores are not normalized to be always positive
        # because masked-out items will not have smallest score (some valid, non-masked ones can be negative)
        # scores = scores * seen_items_mask[user_idx]
        # So instead we do scores = scores * seen_items_mask[user_idx] + NEG_INF * (1 - seen_items_mask[user_idx])
        min_score = scores.min()
        # Unlike in predict_with_score, here we do not mandate NEG_INF to be strictly smaller
        # because rel_scores may already contain some NEG_INF that was set by predict_with_score
        # called previously -> so we allow <=.
        assert RLPropAggregator.NEG_INF <= min_score, f"min_score ({min_score}) is not smaller than NEG_INF ({RLPropAggregator.NEG_INF})"
        scores = scores * seen_items_mask + RLPropAggregator.NEG_INF * (1 - seen_items_mask)
        return scores

    def fast_impl_v3(self, group_ratings_rm, recommendations_number):
        group_size, n_items = group_ratings_rm.shape
        weights = np.ones(shape=(group_size, ), dtype=np.float32)
        weights /= weights.sum()

        # Already in the input, there could be some NEG_INF entries, these needs to be masked out before normalization
        # since the normalization ranges would be affected
        neg_infs = group_ratings_rm <= RLPropAggregator.NEG_INF
        neg_infs = np.repeat(np.any(neg_infs, axis=0, keepdims=True), group_size, axis=0)

        group_ratings_rm[neg_infs] = 0.0 # This is done to mimick original pandas based implementation

        if self.with_norm:
            rating_matrix_scaled = QuantileTransformer().fit_transform(group_ratings_rm.T).T
        else:
            rating_matrix_scaled = group_ratings_rm

        if self.with_norm:
            assert np.all(rating_matrix_scaled >= 0.0), "We expect all mgains to be normalized to be greater than 0"

        # Algorithm variables
        TOT = 0.0
        gm = np.zeros(shape=(group_size, ), dtype=np.float32)
        tots = np.zeros(shape=(n_items, ), dtype=np.float32)

        seen_items_mask = np.ones(shape=(rating_matrix_scaled.shape[1], ), dtype=np.float32)
        # Already masked out by the input
        seen_items_mask[neg_infs[0]] = 0

        top_k = []

        # Precompute the masks outside of the loop
        positive_gain_mask = rating_matrix_scaled >= 0.0
        negative_gain_mask = rating_matrix_scaled < 0.0

        for _ in range(recommendations_number):

            # Added second dimension to enable broadcasting
            gain_items = np.zeros_like(rating_matrix_scaled)

            tots = np.maximum(TOT, TOT + rating_matrix_scaled.sum(axis=0)) # Old, original way

            remainder = tots * weights[:, np.newaxis] - gm[:, np.newaxis]

            assert remainder.shape == (group_size, n_items), f"Need to ensure proper remainder size"

            gain_items[positive_gain_mask] = np.maximum(0, np.minimum(rating_matrix_scaled, remainder)[positive_gain_mask])
            gain_items[negative_gain_mask] = np.minimum(0, (rating_matrix_scaled - remainder)[negative_gain_mask])

            scores = self.mask_scores(gain_items.sum(axis=0), seen_items_mask) # Old WAY
            i_best = scores.argmax()
            seen_items_mask[i_best] = 0

            gm = gm + rating_matrix_scaled[:, i_best]
            TOT = np.clip(gm, 0.0, None).sum()

            top_k.append(i_best)

        return top_k

    # Returns single item recommended at a given time
    def generate_group_recommendations_for_group(self, group_ratings, recommendations_number, **kwargs):
        res3 = self.fast_impl_v3(kwargs["group_ratings_rm"], recommendations_number)
        return {"RLProp": res3}