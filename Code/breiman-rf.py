import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIG
# -----------------------------
N_TREES = 100
N_RUNS = 60
MAX_FEATURES_RANGE = 25

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


# -----------------------------
# CLEAN FEATURES (UNCHANGED)
# -----------------------------
def clean_features(X):
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean(numeric_only=True))
    return X


# -----------------------------
# LOAD DATA (UNCHANGED)
# -----------------------------
def load_dataset(name):

    if name == "sonar":
        df = pd.read_csv("sonar.csv", header=None)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X = clean_features(X)
        y = LabelEncoder().fit_transform(y)

        return X, y

    elif name == "breast":
        df = pd.read_csv("breast.csv", header=None)

        df = df.drop(columns=[0])
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        X = clean_features(X)
        y = LabelEncoder().fit_transform(y)

        return X, y

    else:
        raise ValueError("Unknown dataset")


# -----------------------------
# DECISION TREE (BREIMAN STYLE)
# -----------------------------
class DecisionTree:
    def __init__(self, max_features):
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        probs = np.bincount(y) / m
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        features = np.random.choice(n, self.max_features, replace=False)

        best_gini = float("inf")
        best_idx, best_thresh = None, None

        for idx in features:
            values = np.unique(X[:, idx])

            for t in values:
                left = y[X[:, idx] <= t]
                right = y[X[:, idx] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gini = (
                    len(left) * self._gini(left)
                    + len(right) * self._gini(right)
                ) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = t

        return best_idx, best_thresh

    def _grow_tree(self, X, y):
        # Pure node
        if len(set(y)) == 1:
            return y[0]

        idx, thresh = self._best_split(X, y)

        if idx is None:
            return np.bincount(y).argmax()

        left_mask = X[:, idx] <= thresh
        right_mask = X[:, idx] > thresh

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.bincount(y).argmax()

        left = self._grow_tree(X[left_mask], y[left_mask])
        right = self._grow_tree(X[right_mask], y[right_mask])

        return (idx, thresh, left, right)

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node

        idx, thresh, left, right = node

        if x[idx] <= thresh:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


# -----------------------------
# BREIMAN RANDOM FOREST
# -----------------------------
class BreimanRF:
    def __init__(self, n_trees, max_features):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        self.trees = []

        for _ in range(self.n_trees):
            indices = np.random.choice(n, n, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTree(self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])

        final = []
        for i in range(X.shape[0]):
            values, counts = np.unique(preds[:, i], return_counts=True)
            final.append(values[np.argmax(counts)])

        return np.array(final)

    def all_tree_preds(self, X):
        return [tree.predict(X) for tree in self.trees]


# -----------------------------
# STRENGTH & CORRELATION
# -----------------------------
def compute_strength_correlation(predictions, y_true):
    preds = np.array(predictions)

    final_pred = []
    for i in range(preds.shape[1]):
        values, counts = np.unique(preds[:, i], return_counts=True)
        final_pred.append(values[np.argmax(counts)])
    final_pred = np.array(final_pred)

    strength = np.mean(final_pred == y_true)

    corrs = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            corrs.append(np.mean(preds[i] == preds[j]))

    correlation = np.mean(corrs) if corrs else 0

    return strength, correlation


# -----------------------------
# MAIN EXPERIMENT
# -----------------------------
def run_experiment(dataset_name):
    print(f"\n===== Running dataset: {dataset_name} =====")

    X, y = load_dataset(dataset_name)

    n_features_total = X.shape[1]
    max_feat = min(MAX_FEATURES_RANGE, n_features_total)

    acc_results = []
    err_results = []
    strength_results = []
    corr_results = []

    for f in range(1, max_feat + 1):
        print(f"Running F = {f}")

        acc_runs = []
        err_runs = []
        strength_runs = []
        corr_runs = []

        for run in range(N_RUNS):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=run
            )

            X_train_np = X_train.values
            X_test_np = X_test.values

            rf = BreimanRF(N_TREES, f)
            rf.fit(X_train_np, y_train)

            y_pred = rf.predict(X_test_np)
            acc = accuracy_score(y_test, y_pred)
            err = 1 - acc

            tree_preds = rf.all_tree_preds(X_test_np)

            strength, corr = compute_strength_correlation(tree_preds, y_test)

            acc_runs.append(acc)
            err_runs.append(err)
            strength_runs.append(strength)
            corr_runs.append(corr)

        acc_results.append(np.mean(acc_runs))
        err_results.append(np.mean(err_runs))
        strength_results.append(np.mean(strength_runs))
        corr_results.append(np.mean(corr_runs))

    # -----------------------------
    # PLOTTING
    # -----------------------------
    x = list(range(1, max_feat + 1))

    plt.figure()
    plt.plot(x, acc_results, label="Accuracy")
    plt.plot(x, err_results, label="Error")
    plt.xlabel("Number of Features")
    plt.ylabel("Value")
    plt.title(f"{dataset_name} - Accuracy & Error")
    plt.legend()
    plt.savefig(f"{GRAPH_DIR}/breiman-rf_{dataset_name}_AE.png")
    plt.close()

    plt.figure()
    plt.plot(x, strength_results, label="Strength")
    plt.plot(x, corr_results, label="Correlation")
    plt.xlabel("Number of Features")
    plt.ylabel("Value")
    plt.title(f"{dataset_name} - Strength & Correlation")
    plt.legend()
    plt.savefig(f"{GRAPH_DIR}/breiman-rf_{dataset_name}_SC.png")
    plt.close()

    print(f"✅ Graphs saved for {dataset_name}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
#    run_experiment("sonar")
    run_experiment("breast")