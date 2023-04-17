class ConjugateGradientCounter:
    def __init__(self):
        self.n_iter = 0

    def __call__(self, x=None):
        self.n_iter += 1

    def reset(self):
        self.n_iter = 0
