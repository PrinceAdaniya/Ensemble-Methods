import numpy as np

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# -----------------------------
# LOAD DATASETS
# -----------------------------
def get_datasets():
    datasets = {}

    # Breast Cancer
    bc = load_breast_cancer()
    datasets["breast"] = (bc.data, bc.target)

    # Sonar
    sonar = fetch_openml(name="sonar", version=1, as_frame=False)
    X_sonar = sonar.data
    y_sonar = LabelEncoder().fit_transform(sonar.target)

    datasets["sonar"] = (X_sonar, y_sonar)

    return datasets


# -----------------------------
# TREE
# -----------------------------
class Tree:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def fit(self, X, y):
        self.tree = self._grow(X, y)

    def _split(self, X):
        f = np.random.randint(X.shape[1])
        values = X[:, f]

        min_v, max_v = values.min(), values.max()
        if min_v == max_v:
            return None, None

        low_val = min_v + self.low * (max_v - min_v)
        high_val = min_v + self.high * (max_v - min_v)

        if low_val >= high_val:
            return None, None

        t = np.random.uniform(low_val, high_val)
        return f, t

    def _grow(self, X, y):
        if np.all(y == y[0]):
            return y[0]

        f, t = self._split(X)
        if f is None:
            return np.bincount(y).argmax()

        left = X[:, f] <= t
        right = ~left

        if not left.any() or not right.any():
            return np.bincount(y).argmax()

        return (f, t,
                self._grow(X[left], y[left]),
                self._grow(X[right], y[right]))

    def _predict_one(self, x, node):
        while isinstance(node, tuple):
            f, t, l, r = node
            node = l if x[f] <= t else r
        return node

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


# -----------------------------
# FOREST
# -----------------------------
class Forest:
    def __init__(self, n_trees, low, high):
        self.n_trees = n_trees
        self.low = low
        self.high = high

    def fit(self, X, y):
        self.trees = []
        n = len(X)

        for _ in range(self.n_trees):
            idx = np.random.randint(0, n, n)
            t = Tree(self.low, self.high)
            t.fit(X[idx], y[idx])
            self.trees.append(t)

    def all_preds(self, X):
        return np.array([t.predict(X) for t in self.trees])


# -----------------------------
# METRIC
# -----------------------------
def strength_corr(preds, y):

    final = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=preds
    )
    strength = np.mean(final == y)

    n_trees = preds.shape[0]
    corr = 0
    count = 0

    for i in range(n_trees):
        matches = (preds[i] == preds[i+1:])
        corr += np.sum(np.mean(matches, axis=1))
        count += len(matches)

    corr = corr / count if count else 0

    return strength, corr


# -----------------------------
# RANGE SEARCH
# -----------------------------
values = np.round(np.arange(0.0, 1.0, 0.05), 2)
ranges = [(l, h) for l in values for h in values if h > l]


# -----------------------------
# MAIN
# -----------------------------
datasets = get_datasets()

for name, (X, y) in datasets.items():

    print(f"\n===== DATASET: {name.upper()} =====")

    best_score = -1
    best_range = None
    best_acc = 0

    for low, high in ranges:

        scores = []
        accs = []

        for run in range(30):  # slightly faster

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=run
            )

            rf = Forest(n_trees=20, low=low, high=high)
            rf.fit(X_tr, y_tr)

            preds = rf.all_preds(X_te)

            y_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=preds
            )

            acc = accuracy_score(y_te, y_pred)

            s, c = strength_corr(preds, y_te)
            score = (s ** 2) / (c + 1e-6)

            scores.append(score)
            accs.append(acc)

        mean_score = np.mean(scores)
        mean_acc = np.mean(accs)

        if mean_score > best_score:
            best_score = mean_score
            best_range = (low, high)
            best_acc = mean_acc

    print(f"🔥 BEST RANGE: {best_range[0]:.2f} – {best_range[1]:.2f}")
    print(f"🎯 ACCURACY: {best_acc:.4f}")
    print(f"📈 SCORE: {best_score:.4f}")