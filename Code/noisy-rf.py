import os
import time   # ✅ ADDED
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# -----------------------------
# CONFIG
# -----------------------------
N_TREES = 100
N_RUNS = 60
MAX_FEATURES_RANGE = 25
N_JOBS = -1
NOISE_RATE = 0.1

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


# -----------------------------
# CLEAN FEATURES
# -----------------------------
def clean_features(X):
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean(numeric_only=True))
    return X


# -----------------------------
# LOAD DATA
# -----------------------------
def load_dataset(name):

    if name == "sonar":
        df = pd.read_csv("sonar.csv", header=None)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    elif name == "breast":
        df = pd.read_csv("breast.csv", header=None)
        df = df.drop(columns=[0])
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

    else:
        raise ValueError("Unknown dataset")

    X = clean_features(X)
    y = LabelEncoder().fit_transform(y)

    return X.values, y


# -----------------------------
# LABEL NOISE
# -----------------------------
def add_label_noise(y, noise_rate):
    y_noisy = y.copy()
    n = len(y)
    num_noisy = int(noise_rate * n)

    indices = np.random.choice(n, num_noisy, replace=False)
    classes = np.unique(y)

    for i in indices:
        current = y_noisy[i]
        possible = classes[classes != current]
        y_noisy[i] = np.random.choice(possible)

    return y_noisy


# -----------------------------
# DECISION TREE
# -----------------------------
class DecisionTree:
    def __init__(self, max_features):
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        indices = np.arange(len(X))
        self.tree = self._grow_tree(X, y, indices)

    def _best_split(self, X, y, indices):
        m = len(indices)
        if m <= 1:
            return None, None

        n_features = X.shape[1]
        features = np.random.choice(n_features, self.max_features, replace=False)

        best_gini = float("inf")
        best_idx, best_thresh = None, None
        num_classes = np.max(y) + 1

        for feat in features:
            sorted_idx = indices[np.argsort(X[indices, feat])]
            y_sorted = y[sorted_idx]

            left_counts = np.zeros(num_classes, dtype=int)
            right_counts = np.bincount(y_sorted, minlength=num_classes)

            left_size = 0
            right_size = m

            for i in range(m - 1):
                c = y_sorted[i]

                left_counts[c] += 1
                right_counts[c] -= 1

                left_size += 1
                right_size -= 1

                if X[sorted_idx[i], feat] == X[sorted_idx[i + 1], feat]:
                    continue

                left_prob = left_counts / left_size
                right_prob = right_counts / right_size

                gini_left = 1 - np.sum(left_prob ** 2)
                gini_right = 1 - np.sum(right_prob ** 2)

                gini = (left_size * gini_left + right_size * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = feat
                    best_thresh = (
                        X[sorted_idx[i], feat] + X[sorted_idx[i + 1], feat]
                    ) / 2

        return best_idx, best_thresh

    def _grow_tree(self, X, y, indices):

        if len(set(y[indices])) == 1:
            return y[indices][0]

        idx, thresh = self._best_split(X, y, indices)

        if idx is None:
            return np.bincount(y[indices]).argmax()

        left_mask = X[indices, idx] <= thresh
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return np.bincount(y[indices]).argmax()

        left = self._grow_tree(X, y, left_indices)
        right = self._grow_tree(X, y, right_indices)

        return (idx, thresh, left, right)

    def _predict_batch(self, X, node):
        if not isinstance(node, tuple):
            return np.full(X.shape[0], node)

        idx, thresh, left, right = node

        left_mask = X[:, idx] <= thresh
        right_mask = ~left_mask

        preds = np.empty(X.shape[0], dtype=int)
        preds[left_mask] = self._predict_batch(X[left_mask], left)
        preds[right_mask] = self._predict_batch(X[right_mask], right)

        return preds

    def predict(self, X):
        return self._predict_batch(X, self.tree)


# -----------------------------
# RANDOM FOREST
# -----------------------------
class BreimanRF:
    def __init__(self, n_trees, max_features):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def _build_tree(self, X, y, seed):
        np.random.seed(seed)
        n = len(X)

        indices = np.random.choice(n, n, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        y_noisy = add_label_noise(y_sample, NOISE_RATE)

        tree = DecisionTree(self.max_features)
        tree.fit(X_sample, y_noisy)

        return tree

    def fit(self, X, y):
        self.trees = Parallel(n_jobs=N_JOBS)(
            delayed(self._build_tree)(X, y, i)
            for i in range(self.n_trees)
        )

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
# METRICS
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
# EXPERIMENT (UPDATED)
# -----------------------------
def run_experiment(dataset_name):
    print(f"\n===== Running dataset: {dataset_name} =====")

    start_time = time.time()

    X, y = load_dataset(dataset_name)

    n_features_total = X.shape[1]
    max_feat = min(MAX_FEATURES_RANGE, n_features_total)

    acc_results, err_results = [], []
    strength_results, corr_results = [], []

    for f in range(1, max_feat + 1):
        print(f"Running F = {f}")

        acc_runs, err_runs = [], []
        strength_runs, corr_runs = [], []

        for run in range(N_RUNS):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=run
            )

            rf = BreimanRF(N_TREES, f)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            tree_preds = rf.all_tree_preds(X_test)
            strength, corr = compute_strength_correlation(tree_preds, y_test)

            acc_runs.append(acc)
            err_runs.append(1 - acc)
            strength_runs.append(strength)
            corr_runs.append(corr)

        acc_results.append(np.mean(acc_runs))
        err_results.append(np.mean(err_runs))
        strength_results.append(np.mean(strength_runs))
        corr_results.append(np.mean(corr_runs))

    # PLOTS
    x = list(range(1, max_feat + 1))

    plt.figure()
    plt.plot(x, acc_results, label="Accuracy")
    plt.plot(x, err_results, label="Error")
    plt.legend()
    plt.savefig(f"{GRAPH_DIR}/NOISY_breiman-rf_{dataset_name}_AE.png")
    plt.close()

    plt.figure()
    plt.plot(x, strength_results, label="Strength")
    plt.plot(x, corr_results, label="Correlation")
    plt.legend()
    plt.savefig(f"{GRAPH_DIR}/NOISY_breiman-rf_{dataset_name}_SC.png")
    plt.close()

    end_time = time.time()

    return {
        "dataset": dataset_name,
        "accuracy": np.mean(acc_results),
        "error": np.mean(err_results),
        "time": end_time - start_time
    }


# -----------------------------
# RUN (FINAL OUTPUT TOGETHER)
# -----------------------------
if __name__ == "__main__":
    overall_start = time.time()

    res1 = run_experiment("sonar")
    res2 = run_experiment("breast")

    overall_end = time.time()

    print("\n================ FINAL RESULTS ================\n")

    print(f"📊 {res1['dataset'].upper()}")
    print(f"Accuracy : {res1['accuracy']:.4f}")
    print(f"Error    : {res1['error']:.4f}")
    print(f"Time     : {res1['time']:.2f} sec\n")

    print(f"📊 {res2['dataset'].upper()}")
    print(f"Accuracy : {res2['accuracy']:.4f}")
    print(f"Error    : {res2['error']:.4f}")
    print(f"Time     : {res2['time']:.2f} sec\n")

    print(f"⏱ TOTAL TIME: {(overall_end - overall_start):.2f} sec") 