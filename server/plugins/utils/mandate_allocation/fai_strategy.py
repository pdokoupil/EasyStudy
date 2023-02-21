import numpy as np

class fai_strategy:
    def __init__(self, _, masking_value, *args):
        self.curr_obj = 1
        self.masking_value = masking_value

    def __call__(self, mask, supports):
        masked_supports_neg = (mask * supports + (~mask) * self.masking_value)
        res = np.argmax(masked_supports_neg[self.curr_obj], axis=1)
        self.curr_obj = (self.curr_obj + 1) % masked_supports_neg.shape[0]
        return res