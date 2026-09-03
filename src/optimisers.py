import numpy as np

class StandardGD:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update_weights(self, W1, W2, nablaW1, nablaW2):
        W1_new = W1 - self.alpha * nablaW1
        W2_new = W2 - self.alpha * nablaW2
        return W1_new, W2_new

class MomentumGD:
    def __init__(self, alpha=0.01, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.V_W1 = None
        self.V_W2 = None

    def update_weights(self, W1, W2, nablaW1, nablaW2):
        if self.V_W1 is None:
            self.V_W1 = np.zeros_like(W1)
            self.V_W2 = np.zeros_like(W2)

        self.V_W1 = self.beta * self.V_W1 + (1 - self.beta) * nablaW1
        self.V_W2 = self.beta * self.V_W2 + (1 - self.beta) * nablaW2

        W1_new = W1 - self.alpha * self.V_W1
        W2_new = W2 - self.alpha * self.V_W2
        return W1_new, W2_new
