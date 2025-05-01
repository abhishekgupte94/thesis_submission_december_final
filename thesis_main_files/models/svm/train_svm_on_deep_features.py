from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class SVMTrainer:
    @staticmethod
    def train(dataset, kernel='rbf', C=1.0, test_size=0.2, random_seed=42):
        """
        Trains a CPU-bound scikit-learn SVM classifier from a PyTorch TensorDataset.
        """
        X = dataset.tensors[0].numpy()
        y = dataset.tensors[1].numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        clf = SVC(kernel=kernel, C=C, probability=True)  # probability=True needed for ROC curves
        clf.fit(X_train, y_train)

        return clf, X_test, y_test
