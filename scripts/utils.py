import numpy as np


def logbin(x, bins):
    """Bin an array with expenentially growing bin sizes.

    Parameters
    ----------
    x : array-like
        Array to bin.
    bins : array-like
        Indices to split `x` with.

    Returns
    -------
    x_b : `numpy.ndarray`
        Log-binned array.
    """
    x_split = np.split(x, bins)
    x_b = np.array([i.mean() for i in x_split])
    return x_b


def interp1d(x_out, x_in, y_in, kind='linear'):
    """
    Interpolation method that can also do nearest-value interpolation.
    """
    if kind == 'linear':
        y_out = np.interp(x_out, x_in, y_in)
    elif kind == 'nearest':
        y_out = np.zeros(x_out.size, dtype=np.object)
        for i, x_value in enumerate(x_out):
            nearest = np.argmin(np.abs(x_in - x_value))
            y_out[i] = y_in[nearest]
    else:
        raise ValueError('invalid interpolation method ' + str(kind))
    return y_out


def load_cf(filename):
    with open(filename, 'r') as f:
        out = np.genfromtxt(
            filename,
            delimiter=',',
            skip_header=2,
            names=True,
            dtype=(float, float, float, '<U20', float, float, float, float)
        )
    return out
