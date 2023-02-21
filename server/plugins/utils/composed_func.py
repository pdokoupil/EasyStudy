class ComposedFunc:

    def __init__(self, filters):
        self.filters = filters
    
    def __call__(self, df_filter, *args, **kwargs):
        tmp = df_filter
        for f in self.filters:
            tmp = f(tmp, *args, **kwargs)
        return tmp