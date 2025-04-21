import numpy as np

def stack_features(npv_features, art_avdf_features):
    return np.hstack((npv_features, art_avdf_features))
