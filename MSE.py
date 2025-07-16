import numpy as np


def multiscale_sample_entropy(time_series, max_scale=10, m=2, r=0.2):
    # -------- Coarse-graining  --------
    def coarse_grain(series, scale):
        N = len(series)
        num_segments = N // scale
        return np.array([
            np.mean(series[i * scale:(i + 1) * scale])
            for i in range(num_segments)
        ])

    # -------- sample_entropy --------
    def sample_entropy(series, m, r):
        N = len(series)
        r *= np.std(series)

        def _phi(m):
            x = np.array([series[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.sum(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
            return np.sum(C) / (N - m + 1)

        try:
            return -np.log(_phi(m + 1) / _phi(m))
        except:
            return np.nan

    # -------- Calculating multi-scale sample entropy --------
    mse = []
    for scale in range(1, max_scale + 1):
        y = coarse_grain(time_series, scale)
        se = sample_entropy(y, m, r)
        mse.append(se)

    return np.array(mse)
