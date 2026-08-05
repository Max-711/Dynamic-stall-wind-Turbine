import numpy as np

def error(t, mine, ref, period):
    n = int(np.floor(t[-1] / period)) - 1
    m = (t >= n * period) & (t < (n + 1) * period)
    e = mine[m] - ref[m]
    rng = ref[m].max() - ref[m].min()
    return {"cycle": n,
            "rms": float(np.sqrt(np.mean(e ** 2))),
            "pct": float(100 * np.sqrt(np.mean(e ** 2)) / rng),
            "max": float(np.abs(e).max()),
            "cl_max": float(mine[m].max()), "cl_max_ref": float(ref[m].max()),
            "cl_min": float(mine[m].min()), "cl_min_ref": float(ref[m].min())}