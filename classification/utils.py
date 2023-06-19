import numpy as np

def reshape_mne_raw_for_lda(X: np.ndarray, verbose=False) -> np.ndarray:
    """
    reshapes raw data for LDA, flattening the last dimension
    """
    if verbose:
        print('Data shape from MNE {}'.format(X.shape))
    X_out = np.moveaxis(X, 1, -1)
    if verbose:
        print('new data shape with sampling prioritized over channels {}'.format(X_out.shape))
    X_out = X_out.reshape([X_out.shape[0], X_out.shape[1] * X_out.shape[2]], order='C')
    if verbose:
        print('Shape for K-fold LDA {}'.format(X_out.shape))
    return X_out