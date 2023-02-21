import numpy as np

class exactly_proportional_fuzzy_dhondt:
    def __init__(self, obj_weights, masking_value, *args):
        self.tot = None
        self.s_r = None
        self.votes = (obj_weights / obj_weights.sum())[:, np.newaxis, np.newaxis] # properly expand dims
        self.masking_value = masking_value

    def __call__(self, mask, supports):
        masked_supports_neg = (mask * supports + (~mask) * self.masking_value)
        masked_supports = mask * supports
        if self.tot is None:
            # Shape should be [num_users, 1]
            self.tot = np.zeros((masked_supports.shape[1]), dtype=np.float64)
            # Shape is [num_parties, num_users]
            self.s_r = np.zeros((masked_supports.shape[0], masked_supports.shape[1]), dtype=np.float64)

        # shape [num_users, num_items]
        tot_items = np.full((1, masked_supports.shape[1], masked_supports.shape[2]), self.tot[:, np.newaxis])
        tot_items += np.sum(masked_supports, axis=0)

        # Shape of e_r should be [num_objs, num_users, num_items]
        e_r = np.maximum(0.0, tot_items * self.votes - self.s_r[..., np.newaxis])
        # Shape should be [num_users, num_items]
        gain_items = np.minimum(masked_supports_neg, e_r).sum(axis=0) #np.minimum(masked_supports, e_r).sum(axis=0)

        # Shape should be [num_users,]
        max_gain_items = np.argmax(gain_items, axis=1)

        self.s_r += np.squeeze(np.take_along_axis(masked_supports, max_gain_items[np.newaxis, :, np.newaxis], axis=2))
        self.tot = self.s_r.sum(axis=0)

        return max_gain_items
