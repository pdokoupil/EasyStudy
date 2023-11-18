### File with common metric definitions ###
import functools
import numpy as np
import scipy

# Standard ILD calculation over given 
class intra_list_diversity:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def __call__(self, rec_list):
        top_k_size = len(rec_list)
        if top_k_size <= 1:
            # Single item does not have any diversity
            return 0.0
        # Otherwise calculate intra-list-diversity over the rating features (distances induced by these ratings)
        return self.distance_matrix[np.ix_(rec_list, rec_list)].sum() / (top_k_size * (top_k_size - 1))

### Binomial diversity and everything that is needed to compute it ###
class binomial_diversity:
    def __init__(self, all_categories, get_item_categories, rating_matrix, alpha, loader_name):
        self.all_categories = set(all_categories)
        self.category_getter = get_item_categories
        self.rating_matrix = rating_matrix
        self.alpha = alpha
        self.loader_name = loader_name
        assert alpha == 0.0, f"Current we only support alpha=0 meaning local proportion"

    def __call__(self, rec_list):
        rec_list_categories = self._get_list_categories(rec_list)
        return self._coverage(rec_list, rec_list_categories) * self._non_red(rec_list, rec_list_categories)

    @functools.lru_cache(maxsize=10000)
    def _n_choose_k(self, N, k):
        return scipy.special.comb(N, k)

    # Calculate for each genre separately
    # N is length of the recommendation list
    # k is number of successes, that is, number of items belonging to that genre
    # For each genre, recommendation list is sequence of bernouli trials, and each item in the list having the genre is considered to be a success
    # Calculate probability of k successes in N trials
    def _binomial_probability(self, N, k, p):
        return self._n_choose_k(N, k) * np.power(p, k) * np.power(1.0 - p, N - k) 

    def _get_list_categories(self, rec_list):
        categories = []
        for item in rec_list:
            categories.extend(self.category_getter(item))
        return set(categories)

    # Global part does not change for different users so we calculate it just once
    # @functools.lru_cache(maxsize=None)
    def _p_g_1(self, g):
        # Denominator is for every user take number of items the user has interacted with
        # which reduces to nonzero entries in the rating_matrix
        x = self.rating_matrix.astype(bool)
        denom = x.sum()
        nom = 0.0
        
        # We iterate over columns to speedup the calculation (if we go per rows, we do m x n calls to category_getter
        # instead of just n calls as is the case in column-wise traversal
        # also since in the end we want to aggregate over all users, we do it via sum
        # instead of naive for-loop
        y = x.sum(axis=0)
        for item, cnt in enumerate(y):
            # Consider only items some user has interacted with that also have the given category
            if cnt > 0 and g in self.category_getter(item):
                # Increase by the number of users who made the interaction with the item
                nom += cnt
        
        # Old, naive version
        # for user_ratings in self.rating_matrix:
            # Take items the given user has interacted with
            # i_u = np.where(user_ratings > 0)[0]
            # Get number of items that the user has interacted with and that have the given genre
            # k_g = len([x for x in i_u if g in self.category_getter(x)])
            # nom += k_g

        p_g_1 = nom / denom
        return p_g_1

    def _p_g_2(self, g):
        return 0.0 # TODO if we switch self.alpha != 0.0


    def __eq__(self, other) -> bool:
        if not isinstance(other, binomial_diversity):
            return False
        
        return self.alpha == other.alpha and self.loader_name == other.loader_name

    def __hash__(self) -> int:
        return self.alpha.__hash__() ^ self.loader_name.__hash__()

    # As long as _p_g_2 returns fixed 0, we can cache this function fully
    # Once this changes, we can only cache _p_g_1
    @functools.lru_cache(maxsize=None)
    def _p_g(self, g):
        assert self.alpha == 0.0, f"Current we only support alpha=0 meaning local proportion"
        return (1.0 - self.alpha) * self._p_g_1(g) + self.alpha * self._p_g_2(g)


    # Coverage as in the Binomial algorithm paper
    def _coverage(self, rec_list, rec_list_categories):
        #rec_list_categories = self._get_list_categories(rec_list)
        not_rec_list_categories = self.all_categories - rec_list_categories

        N = len(rec_list)
        
        prod = 1
        for g in not_rec_list_categories:
            p = self._p_g(g)
            prod *= np.power(self._binomial_probability(N, 0, p), 1.0 / len(self.all_categories))

        return prod
    
    # Corresponds to conditional probability used in formulas (10) and (11) in original paper
    def _category_redundancy(self, g, k_g, N):
        s = 0.0
        for l in range(1, k_g):
            # We want P(x_g = l | X_g > 0) so rewrite it as P(x_g = l & X_g > 0) / P(X_g > 0)
            # P(x_g = l & X_g > 0) happens when P(x_g = l) is it already imply X_g > 0
            # so we further simplify this as P(x_g = l) / P(X_g > 0) and P(X_g > 0) can be set to 1 - P(X_g = 0)
            # so we end up with
            # P(x_g = l) / (1 - P(X_g = 0))
            p = self._p_g(g)
            s += (self._binomial_probability(N, l, p) / (1.0 - self._binomial_probability(N, 0, p)))

        return 1.0 - s
    
    def _non_red(self, rec_list, rec_list_categories):
        #rec_list_categories = self._get_list_categories(rec_list)

        N = len(rec_list)
        N_LIST_CATEGORIES = len(rec_list_categories)

        prod = 1.0
        for g in rec_list_categories:
            #num_movies_with_genre = get_num_movies_with_genre(rec_list, g)
            k_g = len([x for x in rec_list if g in self.category_getter(x)])
            p_cond = self._category_redundancy(g, k_g, N)
            prod *= np.power(p_cond, 1.0 / N_LIST_CATEGORIES)

        return prod


