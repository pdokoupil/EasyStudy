import numpy as np

class exactly_proportional_fuzzy_dhondt_2:
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
        tot_items += np.sum(np.maximum(0.0, masked_supports), axis=0)

        # Shape of e_r should be [num_objs, num_users, num_items]
        unused_p = tot_items * self.votes - self.s_r[..., np.newaxis]
        
        positive_support_mask = masked_supports_neg >= 0.0
        negative_support_mask = masked_supports_neg < 0.0
        gain_items = np.zeros_like(masked_supports, dtype=np.float64)
        gain_items[positive_support_mask] = np.maximum(0, np.minimum(masked_supports_neg[positive_support_mask], unused_p[positive_support_mask]))
        #np.put(gain_items, positive_support_mask, np.maximum(0, np.minimum(np.take(masked_supports, positive_support_mask), unused_p)))
        gain_items[negative_support_mask] = np.minimum(0, masked_supports_neg[negative_support_mask] - unused_p[negative_support_mask])
        #np.put(gain_items, negative_support_mask, np.minimum(0, np.take(masked_supports, negative_support_mask) - unused_p))

        # Shape should be [num_users, num_items]
        gain_items = gain_items.sum(axis=0)

        # Shape should be [num_users,]
        #max_gain_items = np.argmax(gain_items, axis=1)
        max_gains = np.max(gain_items, axis=1, keepdims=True)
        #all_max_gain_items = np.where(gain_items < max_gains)
        # Break ties by selecting items with maximal tot_items
        max_gain_items = np.argmax(np.where(gain_items == max_gains, np.squeeze(tot_items), np.NINF), axis=1)

        self.s_r += np.squeeze(np.take_along_axis(masked_supports, max_gain_items[np.newaxis, :, np.newaxis], axis=2), axis=2)
        self.tot = np.where(self.s_r >= 0, self.s_r, 0).sum(axis=0) # TODO NEW
        # self.tot = self.s_r.sum(axis=0)

        return max_gain_items
