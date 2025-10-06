import time

class OneEuroFilter:
    """Simple One Euro filter for smoothing with minimal latency.

    Based on: https://cristal.univ-lille.fr/~casiez/1euro/
    """

    def __init__(self, freq=120.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2.0 * 3.1415926535 * cutoff)
        te = 1.0 / max(1e-6, self.freq)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        t = timestamp if timestamp is not None else time.time()
        if self.t_prev is not None and t != self.t_prev:
            self.freq = 1.0 / (t - self.t_prev)
        self.t_prev = t

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # Derivative of the signal
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

