import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import time


# FIX RANDOMNESS

np.random.seed(42)


# LOAD DATA

print("Loading data...")

df = pd.read_csv(
    r"C:\Users\yashr\OneDrive\Desktop\prjt aiml\HIGGS.csv",
    header=None,
    nrows=500000
)

y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values.astype(np.float32)


# TREE

class Tree:
    def __init__(self, mode="context", max_depth=4, reg_lambda=1.0, gamma=0.0, n_bins=32, top_k=10):
        self.mode = mode
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.top_k = top_k

    def _gain(self, GL, HL, GR, HR):
        return (GL**2)/(HL+self.reg_lambda) + (GR**2)/(HR+self.reg_lambda) - ((GL+GR)**2)/(HL+HR+self.reg_lambda) - self.gamma

    def _leaf(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.reg_lambda)

    def fit(self, X, g, h):
        self.root = self._build(X, g, h, 0)

    def _build(self, X, g, h, depth):
        if depth >= self.max_depth or len(g) < 200:
            return self._leaf(g, h)

        n, m = X.shape
        best_gain = -1
        best = None

        scores = []
        for j in range(m):
            col = X[:, j]
            valid = ~np.isnan(col)
            if np.sum(valid) < 50:
                continue
            score = np.abs(np.mean(col[valid] * g[valid]))
            scores.append((score, j))

        scores.sort(reverse=True)
        features = [j for _, j in scores[:self.top_k]]

        for j in features:
            col = X[:, j]
            valid = ~np.isnan(col)
            missing = np.isnan(col)

            if np.sum(valid) == 0:
                continue

            col_valid = col[valid]
            g_valid = g[valid]
            h_valid = h[valid]

            bins = np.linspace(col_valid.min(), col_valid.max(), self.n_bins)
            ids = np.digitize(col_valid, bins)

            G = np.bincount(ids, weights=g_valid, minlength=self.n_bins+1)
            H = np.bincount(ids, weights=h_valid, minlength=self.n_bins+1)

            Gp = np.cumsum(G)
            Hp = np.cumsum(H)

            G_total = Gp[-1]
            H_total = Hp[-1]

            G_nan = np.sum(g[missing])
            H_nan = np.sum(h[missing])

            for b in range(self.n_bins):
                GL, HL = Gp[b], Hp[b]
                GR, HR = G_total-GL, H_total-HL

                if self.mode == "default":
                    gain_left = self._gain(GL+G_nan, HL+H_nan, GR, HR)
                    gain_right = self._gain(GL, HL, GR+G_nan, HR+H_nan)

                    if gain_left > gain_right:
                        gain = gain_left
                        direction = "left"
                    else:
                        gain = gain_right
                        direction = "right"

                    if gain > best_gain:
                        best_gain = gain
                        best = (j, bins[b], direction)

                else:
                    gain = self._gain(GL, HL, GR, HR)
                    if gain > best_gain:
                        best_gain = gain
                        best = (j, bins[b])

        if best is None:
            return self._leaf(g, h)

        # DEFAULT
        if self.mode == "default":
            j, thr, direction = best
            col = X[:, j]

            if direction == "left":
                left = (col <= thr) | np.isnan(col)
                right = (col > thr)
            else:
                left = (col <= thr)
                right = (col > thr) | np.isnan(col)

            return {
                "f": j, "t": thr, "dir": direction,
                "l": self._build(X[left], g[left], h[left], depth+1),
                "r": self._build(X[right], g[right], h[right], depth+1)
            }

        # CONTEXT
        j, thr = best
        col = X[:, j]

        best_ctx, best_score = None, -1

        for k in features:
            if k == j:
                continue
            ctx = X[:, k]
            valid = ~np.isnan(ctx)
            if np.sum(valid) < 50:
                continue

            score = np.abs(np.mean(ctx[valid] * g[valid]))
            if score > best_score:
                best_score = score
                best_ctx = k

        if best_ctx is None:
            best_ctx = j

        ctx_col = X[:, best_ctx]
        ctx_thr = np.nanmedian(ctx_col)

        left, right = [], []

        for i in range(n):
            v = col[i]
            if np.isnan(v):
                ctx_val = ctx_col[i]
                go_left = True if np.isnan(ctx_val) else ctx_val <= ctx_thr
            else:
                go_left = v <= thr

            (left if go_left else right).append(i)

        left, right = np.array(left), np.array(right)

        if len(left) == 0 or len(right) == 0:
            return self._leaf(g, h)

        return {
            "f": j, "t": thr,
            "ctx_f": best_ctx, "ctx_t": ctx_thr,
            "l": self._build(X[left], g[left], h[left], depth+1),
            "r": self._build(X[right], g[right], h[right], depth+1)
        }

    def _pred_row(self, x, node):
        if not isinstance(node, dict):
            return node

        v = x[node["f"]]

        if "dir" in node:
            if np.isnan(v):
                return self._pred_row(x, node["l"] if node["dir"]=="left" else node["r"])
        else:
            if np.isnan(v):
                ctx = x[node["ctx_f"]]
                if np.isnan(ctx) or ctx <= node["ctx_t"]:
                    return self._pred_row(x, node["l"])
                else:
                    return self._pred_row(x, node["r"])

        if v <= node["t"]:
            return self._pred_row(x, node["l"])
        else:
            return self._pred_row(x, node["r"])

    def predict(self, X):
        return np.array([self._pred_row(x, self.root) for x in X])



# MODEL (100 TREES)

class Model:
    def __init__(self, mode, n_estimators=200):
        self.mode = mode
        self.n_estimators = n_estimators
        self.trees = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        preds = np.zeros(len(y))

        for i in range(self.n_estimators):
            print(f"{self.mode} Tree {i+1}/{self.n_estimators}")

            p = self._sigmoid(preds)
            g = p - y
            h = p * (1 - p)

            tree = Tree(mode=self.mode)
            tree.fit(X, g, h)

            preds += 0.05 * tree.predict(X)
            self.trees.append(tree)

    def predict_proba(self, X):
        preds = np.zeros(len(X))
        for t in self.trees:
            preds += 0.05 * t.predict(X)
        return self._sigmoid(preds)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)



# EXPERIMENT (10 FEATURES)


np.random.seed(42)

#  CHANGE: now 10 columns
cols = np.random.choice(X.shape[1], 10, replace=False)

print("Selected columns for missing:", cols)

for miss_ratio in [0.1, 0.3, 0.5]:

    print("\n==============================")
    print(f"Missing {int(miss_ratio*100)}% (PER COLUMN)")
    print("==============================")

    X_copy = X.copy()

    n = X.shape[0]
    k = int(miss_ratio * n)

    # Inject missing values
    for col in cols:
        idx = np.random.choice(n, k, replace=False)
        X_copy[idx, col] = np.nan

    # Info prints
    print("Columns affected:", len(cols))
    print("Per-column missing:", miss_ratio)
    print("Actual overall missing:", round(np.isnan(X_copy).mean(), 4))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_copy, y, test_size=0.2, random_state=42
    )

    # Run both methods
    for mode in ["default", "context"]:
        print(f"\nRunning {mode}")

        model = Model(mode=mode, n_estimators=200)

        start = time.time()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("AUC:", round(roc_auc_score(y_test, y_prob), 4))
        print("Time:", round(time.time() - start, 2), "sec")