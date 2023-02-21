from sklearn.preprocessing import RobustScaler

class robust_scaler:
    def __init__(self, shift=0.0):
        self.transformer = RobustScaler()
        self.shift = shift

    def __call__(self, supports, ignore_shift=False):
        if ignore_shift:
            return self.transformer.transform(supports)
        # supports have shape [num_users, num_data_points] or [num_data_points]
        return self.transformer.transform(supports) + self.shift

    def train(self, data_points):
        self.transformer.fit(data_points)