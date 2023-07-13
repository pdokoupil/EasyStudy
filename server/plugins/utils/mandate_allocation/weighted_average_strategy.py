import numpy as np

class weighted_average_strategy:
    def __init__(self, obj_weights, masking_value, *args):
        self.obj_weights = obj_weights[:, np.newaxis, np.newaxis]
        self.masking_value = masking_value

    def __call__(self, mask, supports):
        masked_supports_neg = (mask * supports + (~mask) * self.masking_value)
        return np.argmax(np.sum(masked_supports_neg * self.obj_weights, axis=0), axis=1)