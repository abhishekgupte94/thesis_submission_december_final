# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from
# def train_svm_on_deep_features(
#     dataset: TensorDataset,
#     kernel: str = 'rbf',
#     C: float = 1.0,
#     test_size: float = 0.2,
#     random_seed: int = 42
# ):
#     """
#     Trains an SVM on concatenated deep features.
#
#     Args:
#         dataset (TensorDataset): Output of build_audio_video_deep_feature_dataset().
#         kernel (str): Kernel type for SVM ('linear', 'rbf', etc.).
#         C (float): Regularization parameter.
#         test_size (float): Proportion for test split.
#         random_seed (int): Random seed.
#
#     Returns:
#         clf (SVC): Trained SVM classifier.
#         X_test, y_test: For later evaluation.
#     """
#     X = dataset.tensors[0].numpy()
#     y = dataset.tensors[1].numpy()
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_seed
#     )
#
#     clf = SVC(kernel=kernel, C=C, probability=False)
#     clf.fit(X_train, y_train)
#
#     return clf, X_test, y_test
