import numpy as np


def _add_noise(rng: np.random.Generator, data: np.ndarray, noise: float) -> np.ndarray:
    return data + rng.uniform(-noise, noise, size=data.shape)


def generate_regression(n: int, seed: int = None):
    """
    Generates a synthetic regression dataset with 4 inputs and 3 continuous outputs.

    Parameters
    ----------
    n : int
        Number of samples to generate.

    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    i1 = rng.uniform(-3, 3, size=n)
    i2 = rng.uniform(-3, 3, size=n)
    i3 = rng.uniform(-3, 3, size=n)
    i4 = rng.uniform(-3, 3, size=n)

    o1 = i1*i4 + 5*i2 - 2*i1*i3 + i4
    o2 = 4*i1 + 2*i2*i3 + 0.4*i4*i2 + 3*i3
    o3 = i1 + 0.3*i2 + 2*i3*i2 + 2*i4

    X = _add_noise(rng, np.column_stack((i1, i2, i3, i4)), noise=0.5)
    Y = np.column_stack((o1, o2, o3))

    return X.tolist(), Y.tolist()


def generate_multilabel(n: int, seed: int = None):
    """
    Generates a synthetic multilabel classification dataset with 4 inputs and 3 binary outputs.

    Parameters
    ----------
    n : int
        Number of samples to generate.

    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    i1 = rng.uniform(-6, 6, size=n)
    i2 = rng.uniform(-6, 6, size=n)
    i3 = rng.uniform(-6, 6, size=n)
    i4 = rng.uniform(-6, 6, size=n)

    o1 = (i1*i4 - 5*i2 < 2*i1*i3 - i4).astype(float)
    o2 = (4*i1 - 2*i2*i3 + 0.4*i4*i2/i1 < -3*i3).astype(float)
    o3 = (-i1/i4 + 0.3*i2 - 8*i2*i2/i3 < 2*i4).astype(float)

    X = _add_noise(rng, np.column_stack((i1, i2, i3, i4)), noise=0.2)
    Y = np.column_stack((o1, o2, o3))

    return X.tolist(), Y.tolist()


def generate_multiclass(n: int, seed: int = None):
    """
    Generates a synthetic multiclass classification dataset with 6 inputs and 3 classes (one-hot).

    Parameters
    ----------
    n : int
        Number of samples to generate.

    seed : int, optional
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    n_inputs  = 6
    n_classes = 3

    X = np.zeros((n, n_inputs))
    Y = np.zeros((n, n_classes))

    class_labels = rng.integers(0, n_classes, size=n)

    class_ranges = {
        0: [(-2, 3), (1, 5), (0, 4), (-3, 5), (-3, 3), (-2, 2)],
        1: [(-1, 4), (-2, 3), (1, 6), (1, 5),  (0, 6),  (-5, 1)],
        2: [(0, 3),  (-2, 2), (-2, 5), (-4, 2), (-2, 4), (-3, 5)],
    }

    for c in range(n_classes):
        idx = np.where(class_labels == c)[0]
        for feature, (low, high) in enumerate(class_ranges[c]):
            X[idx, feature] = rng.uniform(low, high, size=len(idx))
        Y[idx, c] = 1.0

    X = _add_noise(rng, X, noise=0.2)

    return X.tolist(), Y.tolist()
