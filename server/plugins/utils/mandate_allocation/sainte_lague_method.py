import numpy as np

class sainte_lague_method:
    def __init__(self, obj_weights, masking_value, *args):
        self.seats = None
        self.obj_wieghts = obj_weights[:, np.newaxis]
        self.masking_value = masking_value
    # Sainte lague method as described in:
    #   Diversity by Proportionality: An Election-based Approach to Search Result Diversification
    # https://ciir-publications.cs.umass.edu/getpdf.php?id=1050
    def __call__(self, mask, supports):
        if self.seats is None:
            # Seats per party and per user
            self.seats = np.zeros((supports.shape[0], supports.shape[1]), dtype=np.float64)
        
        masked_supports_neg = (mask * supports + (~mask) * self.masking_value)
        # Shape should be [num_parties, num_users]
        per_party_quotient = self.obj_wieghts / (2.0 * self.seats + 1)
        # Shape should be [num_users]
        selected_parties = np.argmax(per_party_quotient, axis=0)

        # Too much memory needed
        # best_items = np.argmax(masked_supports_neg[selected_parties], axis=1)

        best_items = np.zeros_like(selected_parties, dtype=np.int32)
        for i, p in enumerate(selected_parties):
            best_items[i] = np.argmax(masked_supports_neg[p, i], axis=0)

        self.seats[selected_parties] += 1
        return best_items