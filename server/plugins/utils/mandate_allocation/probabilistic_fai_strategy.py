import numpy as np

class probabilistic_fai_strategy:
    def __init__(self, obj_weights, masking_value, *args):
        self.obj_weights = obj_weights
        self.masking_value = masking_value

    # supports.shape[0] corresponds to number of objectives
    def __call__(self, mask, supports):
        masked_supports_neg = (mask * supports + (~mask) * self.masking_value)
        curr_obj = np.random.choice(np.arange(masked_supports_neg.shape[0]), p=self.obj_weights)
        return np.argmax(masked_supports_neg[curr_obj], axis=1)