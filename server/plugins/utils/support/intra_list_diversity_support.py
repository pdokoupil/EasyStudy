import numpy as np
# old_obj_values are diversityy values for lists of length k - 1
def intra_list_diversity_support(users_partial_lists, items, distance_matrix, k):
    # return ((old_obj_values * (k - 1)) + (distance_matrix[users_partial_list, item].sum() * 2)) / k

    # diversities between single top_k list an array of items (so per user)
    # distance_matrix[indices[:,np.newaxis], item_indices].sum(axis=0)

    # for multiple users we just work with rank 3 tensors .. results is 2d where first axis is user
    # expanding dims via newaxis is needed, alternative is to use d[rows][:,cols] syntax or np.ix_, however, both seems
    # to be slower
    #return ((old_obj_values * (k - 1)) + (distance_matrix[top_k_lists[:,:,np.newaxis], [1, 3]].sum(axis=1) * 2)) / k
    
    if k == 1:
        return np.repeat(np.expand_dims(distance_matrix.sum(axis=1), axis=0), users_partial_lists.shape[0], axis=0) / (distance_matrix.shape[0] - 1)
    elif k == 2:
        old_supp = distance_matrix[users_partial_lists[:, 0]].sum(axis=1, keepdims=True) / (distance_matrix.shape[0] - 1) # TODO remove, probably not necessary as it is constant per user
        return 2 * distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / k - old_supp
        
    return distance_matrix[users_partial_lists[:,:k-1,np.newaxis], items].sum(axis=1) / (k - 1)
