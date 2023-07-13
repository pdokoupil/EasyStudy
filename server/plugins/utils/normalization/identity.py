# Identity normalization == no normalization
class identity:
    def __init__(self, *args):
        pass

    def __call__(self, supports, *args):
        return supports

    def train(self, *args):
        pass