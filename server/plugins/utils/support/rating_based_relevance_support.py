# Returns np.array with shape num_users x num_items which has meaning of support of item for user at step k
def rating_based_relevance_support(rating_matrix):
    return rating_matrix