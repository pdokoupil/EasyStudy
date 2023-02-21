

class SubtractMeanNormalize:
    # Normalize the rating matrix by subtracting mean rating of each user
    def __call__(self, rating_matrix):
        return rating_matrix - rating_matrix.mean(axis=1, keepdims=True)