"""Kumo Tabular con ONNX Runtime: download del modello, preprocessing (porting numpy
della ricetta NVIDIA sdm/models/kumo/tabular/recipe.py) e stimatore KumoTabularONNX."""

import hashlib
import math
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# ============================================================
# Download del modello .onnx
# ============================================================
REPO_URL = "https://huggingface.co/mrfakename/Kumo-Tabular-ONNX/resolve/main/onnx/"
MODELS = {  # nome file, sha256
    "classification": ("small_classification.onnx", "168e2cb32b92a915d28bd7fdf59c34ae32bac656f90abeb58b7a777bc74329f8"),
    "regression": ("small_regression.onnx", "0d792ad9658ee2b6f534eca83522bff209f94bb7ad8d838ca53b3c86df906fff"),
}
MODEL_DIR = Path(__file__).resolve().parent / "models"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(task):
    name, expected = MODELS[task]
    path = MODEL_DIR / name
    if path.exists() and sha256(path) == expected:
        return path
    MODEL_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(REPO_URL + name) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            print(f"\r{name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
    print()
    if sha256(tmp) != expected:
        tmp.unlink()
        raise RuntimeError(f"Checksum errato per {name}")
    tmp.replace(path)
    return path


# ============================================================
# Preprocessing
# ============================================================
EPS = np.finfo(np.float64).eps
F64_MAX = np.finfo(np.float64).max
F64_TINY = np.finfo(np.float64).tiny
MISSING = {"", "nan", "NaN", "NAN", "null", "NULL", "None", "NA", "N/A", "n/a", "?", "<NA>", "NaT"}
MAX_COLUMNS, COUNT_MIN_CARDINALITY, MAX_CLASSES = 500, 50, 10
YJ_LOG_TINY = (math.log(F64_TINY) - math.log(EPS)) / 2  # limiti di lambda (sdm power.py)
YJ_LOG_MAX = (math.log(F64_MAX) + math.log(EPS)) / 2

# ---- funzioni normali (CDF e inversa) senza scipy
_erfc = np.vectorize(math.erfc, otypes=[float])


def ndtr(x):
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def ndtri(p):
    # Algoritmo di Acklam + un passo di Halley (errore ~1e-15)
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    with np.errstate(all="ignore"):
        ql = np.sqrt(-2 * np.log(p))
        xl = (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
        q = p - 0.5
        r = q * q
        xm = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        qh = np.sqrt(-2 * np.log(1 - p))
        xh = -(((((c[0] * qh + c[1]) * qh + c[2]) * qh + c[3]) * qh + c[4]) * qh + c[5]) / ((((d[0] * qh + d[1]) * qh + d[2]) * qh + d[3]) * qh + 1)
        x = np.where(p < 0.02425, xl, np.where(p <= 1 - 0.02425, xm, xh))
        u = (ndtr(x) - p) * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
        x = np.where(np.abs(x) < 6, x - u / (1 + x * u / 2), x)
    x = np.where(p == 0, -np.inf, np.where(p == 1, np.inf, x))
    return np.where((p < 0) | (p > 1) | np.isnan(p), np.nan, x)


# ---- statistiche sulle prime n righe (contesto), solo valori finiti
def _finite_ctx(col, n):
    c = col[:n]
    return c[np.isfinite(c)]


def _pop_stats(col, n):
    f = _finite_ctx(col, n)
    if f.size == 0:
        return 0.0, 0.0, 0
    m = f.mean()
    return m, ((f - m) ** 2).mean(), f.size


def _is_const(mean, var, k):
    return var <= k * EPS * var + (k * mean * EPS) ** 2


def _keep_non_constant(col, n):
    if n <= 1:
        return True
    c = col[:n]
    return bool(np.any(c != c[0])) and not bool(np.isnan(c).all())


def _standardize_clip(col, n):
    m, v, _ = _pop_stats(col, n)
    with np.errstate(invalid="ignore"):
        return np.clip((col - m) / (math.sqrt(v) + 1e-6), -100.0, 100.0)


# ---- i 4 rami
def _yeo_johnson(x, lam):
    out = np.full_like(x, np.nan)
    pos, neg = x >= 0, x < 0
    with np.errstate(all="ignore"):
        out[pos] = np.log1p(x[pos]) if abs(lam) < EPS else np.expm1(lam * np.log1p(x[pos])) / lam
        m = 2.0 - lam
        out[neg] = -np.log1p(-x[neg]) if abs(m) < EPS else -np.expm1(m * np.log1p(-x[neg])) / m
    return out


def _power_transform(col, n):
    fit = _finite_ctx(col, n)
    nn = max(fit.size, 1)
    mean, var, k = _pop_stats(col, n)
    lam = 1.0
    if fit.size and not _is_const(mean, var, k):
        A = np.abs(fit).max()
        if A == 0:
            lo = hi = 1.0
        else:
            L = math.log1p(20 * A)
            lo, hi = YJ_LOG_TINY / L, YJ_LOG_MAX / L
            if (fit < 0).all():
                lo, hi = 2 - hi, 2 - lo
            elif (fit < 0).any():
                lo = max(2 - hi, lo)
                hi = min(2 - lo, hi)
        log_term = float(np.sum(np.sign(fit) * np.log1p(np.abs(fit))))

        def loglik(l):
            with np.errstate(all="ignore"):
                t = _yeo_johnson(fit, l)
                v = ((t - t.sum() / nn) ** 2).sum() / nn
                if not np.isfinite(v) or v < F64_TINY:
                    return -np.inf
                return -nn / 2 * math.log(v) + (l - 1) * log_term

        if lo == hi:
            lam = lo
        else:  # golden section search, 44 passi come sdm
            ip = (math.sqrt(5) - 1) / 2
            l, r = lo, hi
            c, d = r - (r - l) * ip, l + (r - l) * ip
            fc, fd = loglik(c), loglik(d)
            for _ in range(44):
                if fc < fd:
                    l, c, fc = c, d, fd
                    d = l + (r - l) * ip
                    fd = loglik(d)
                else:
                    r, d, fd = d, c, fc
                    c = r - (r - l) * ip
                    fc = loglik(c)
            lam = (l + r) / 2
    t = _yeo_johnson(col, lam)
    m2, v2, k2 = _pop_stats(t, n)
    scale = 1.0 if _is_const(m2, v2, k2) else math.sqrt(v2)
    with np.errstate(all="ignore"):
        return np.clip((t - m2) / scale, -F64_MAX, F64_MAX)


def _robust_scale_soft_clip(col, n):
    f = _finite_ctx(col, n)
    med, scale = 0.0, 1.0
    if f.size:
        q1, med, q3 = np.quantile(f, [0.25, 0.5, 0.75])
        scale = 1.0 if q1 == q3 else q3 - q1
    with np.errstate(all="ignore"):
        v = (col - med) / scale
        u = np.abs(v) / 3
        s = np.sqrt(1 + u * u)
        return np.sign(v) * 3 * np.where(np.isfinite(s), u / s, 1.0)


def _rank_gaussian(col, n, max_knots=8192):
    v = np.sort(_finite_ctx(col, n))
    cnt = v.size
    if cnt == 0:
        return np.full_like(col, np.nan)
    p = np.minimum((np.searchsorted(v, v, "left") + np.searchsorted(v, v, "right")) / (2 * cnt), 1 - EPS)
    if n <= max_knots:
        knots, kp = v, p
    else:
        idx = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
        if idx.size > max_knots:
            lo, hi = ndtri(p[0]), ndtri(p[-1])
            q = ndtr(lo + (hi - lo) * np.arange(max_knots) / (max_knots - 1))
            a = np.minimum(np.searchsorted(p, q, "left"), cnt - 1)
            b = np.maximum(a - 1, 0)
            idx = np.where(q - p[b] < p[a] - q, b, a)
            idx[0], idx[-1] = 0, cnt - 1
        knots, kp = v[idx], p[idx]
    K = knots.size
    with np.errstate(all="ignore"):
        if K == 1:
            P = np.full_like(col, kp[0])
        else:
            i = np.clip(np.searchsorted(knots, col, "right"), 1, K - 1)
            den = knots[i] - knots[i - 1]
            w = np.where(den == 0, 0.0, (col - knots[i - 1]) / np.where(den == 0, 1.0, den))
            P = kp[i - 1] + (kp[i] - kp[i - 1]) * w
            P = np.where(col <= knots[0], kp[0], np.where(col >= knots[-1], kp[-1], P))
        out = ndtri(P)
    out[np.isnan(col)] = np.nan
    return out


def _clip_sigma(col, n, thr=4.0):
    def sample_stats(vals):
        if vals.size == 0:
            return None
        m = vals.mean()
        return m, max(math.sqrt(((vals - m) ** 2).sum() / max(vals.size - 1, 1)), 1e-6)

    f = _finite_ctx(col, n)
    m, s = sample_stats(f) or (0.0, 1e-6)
    m, s = sample_stats(f[(f >= m - thr * s) & (f <= m + thr * s)]) or (m, s)
    lo, hi = m - thr * s, m + thr * s
    with np.errstate(invalid="ignore"):
        l = np.log1p(np.abs(col))
        return np.minimum(l + hi, np.maximum(lo - l, col))


BRANCHES = [lambda c, n: c, _power_transform, _robust_scale_soft_clip, _rank_gaussian]


# ---- tipi di colonna (senza pandas)
def _missing(v):
    v = np.asarray(v)
    if v.dtype.kind in "fc":
        return np.isnan(v)
    if v.dtype.kind == "M":
        return np.isnat(v)
    if v.dtype.kind in "iub":
        return np.zeros(len(v), dtype=bool)
    return np.array([x is None or (isinstance(x, float) and math.isnan(x)) or str(x).strip() in MISSING for x in v], dtype=bool)


def _try_float(x):
    try:
        return float(str(x).strip())
    except ValueError:
        return None


def _is_numeric_column(v):
    k = v.dtype.kind
    if k == "b":
        return False
    if k in "iufM":
        return True
    vals = v[~_missing(v)]
    return vals.size > 0 and all(not isinstance(x, (bool, np.bool_)) and _try_float(x) is not None for x in vals)


def _numeric_values(v):
    k = v.dtype.kind
    if k == "M":
        out = v.astype("datetime64[s]").astype("int64").astype(float)  # secondi
        out[np.isnat(v)] = np.nan
        return out
    if k in "iufb":
        return v.astype(float)
    miss = _missing(v)
    out = np.array([np.nan if m else _try_float(x) for x, m in zip(v, miss)], dtype=object)
    out[np.equal(out, None)] = np.nan
    return out.astype(float)


class Features:
    # Colonne preprocessate (contesto + query) e rami in cache.
    def __init__(self, ctx, qry, n, categorical=()):
        self.n = n
        num_cols, cat_cols = [], []
        for f, v in ctx.items():
            (cat_cols if f in categorical or not _is_numeric_column(v) else num_cols).append(f)
        allv = {f: np.concatenate([np.asarray(ctx[f], dtype=object if ctx[f].dtype != qry[f].dtype else None), qry[f]])
                for f in ctx}

        count_cols, id_cols = [], []
        for f in cat_cols:
            miss = _missing(allv[f])
            s = [str(x) for x in allv[f]]
            vocab = sorted({x for x, m in zip(s[:n], miss[:n]) if not m})
            index = {x: i for i, x in enumerate(vocab)}
            codes = np.array([-1.0 if m else float(index.get(x, -1)) for x, m in zip(s, miss)])
            if len(vocab) > COUNT_MIN_CARDINALITY:
                slot = np.where(codes < 0, len(vocab), codes).astype(int)
                counts = np.bincount(slot[:n], minlength=len(vocab) + 1).astype(float)
                count_cols.append((f"{f}__count", np.log1p(counts[slot])))
            id_cols.append((f, codes))

        self.columns = []  # (nome, is_categorical, flip, valori_base)
        for f in num_cols:
            raw = _numeric_values(allv[f])
            if _keep_non_constant(raw, n):
                self.columns.append((f, False, True, _standardize_clip(raw, n)))
        for name, c in count_cols:
            if _keep_non_constant(c, n):
                self.columns.append((name, False, False, _standardize_clip(c, n)))
        for name, c in id_cols:
            if _keep_non_constant(c, n):
                self.columns.append((name, True, False, _standardize_clip(c, n)))
        self._cache = {}

    def branch(self, b):
        if b not in self._cache:
            self._cache[b] = [_clip_sigma(BRANCHES[b](c[3], self.n), self.n) for c in self.columns]
        return self._cache[b]

    def build_members(self, E, rng):
        # Un input (x, mask) per stimatore: FlipSign + ShuffleColumns(latin) + SelectColumns(500)
        F = len(self.columns)
        signs = [[(-1.0 if rng.random() < 0.5 else 1.0) if c[2] else 1.0 for c in self.columns] for _ in range(E)]
        base, rows_p = rng.permutation(F), rng.permutation(F)
        members = []
        for m in range(E):
            perm = np.arange(F) if F <= 1 else base[((m % F) - rows_p) % F]
            sel = perm[:MAX_COLUMNS]
            cols = self.branch(m % 4)
            x = np.stack([cols[ci] * signs[m][ci] for ci in sel], axis=1).astype(np.float32)
            mask = np.array([self.columns[ci][1] for ci in sel], dtype=bool)
            members.append((x, mask))
        return members


# ============================================================
# Modello Kumo Tabular con ONNX Runtime
# ============================================================
_SESSIONS = {}  # cache: il file .onnx si carica una volta sola


def get_session(task, providers=None):
    if providers is None:  # GPU se disponibile, altrimenti CPU
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in ort.get_available_providers()]
    key = (task, tuple(providers))
    if key not in _SESSIONS:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[key] = ort.InferenceSession(str(download_model(task)), so, providers=providers)
    return _SESSIONS[key]


def as_columns(X, names=None):
    if isinstance(X, dict):
        return {str(k): np.asarray(v) for k, v in X.items()}
    if hasattr(X, "columns"):  # DataFrame pandas
        return {str(c): X[c].to_numpy() for c in X.columns}
    X = np.asarray(X)
    X = X[:, None] if X.ndim == 1 else X
    names = names or [f"f{i}" for i in range(X.shape[1])]
    return {name: X[:, i] for i, name in enumerate(names)}


class KumoTabularONNX:
    def __init__(self, task="classification", n_estimators=8, seed=0, categorical_features=(),
                 providers=None, chunk_size=256):
        assert task in ("classification", "regression")
        self.task, self.n_estimators, self.seed = task, n_estimators, seed
        self.categorical_features = set(categorical_features)
        self.chunk_size = chunk_size
        self.session = get_session(task, providers)

    def set_context(self, X, y):
        # Nessun training: i pesi .onnx sono pre-addestrati e fissi. Qui si memorizzano solo le
        # righe etichettate (il "contesto") che il modello leggerà in predict, come gli esempi in un prompt.
        cols = as_columns(X)
        y = np.asarray(y)
        ok = ~_missing(y)
        self._X = {f: v[ok] for f, v in cols.items()}
        y = y[ok]
        if len(y) < 2:
            raise ValueError("Servono almeno 2 righe di contesto con target")
        if self.task == "classification":
            if y.dtype.kind in "iufb":
                self.classes_ = np.unique(y)
                self._y = np.searchsorted(self.classes_, y).astype(np.int64)
            else:
                ys = np.array([str(v) for v in y])
                self.classes_ = np.unique(ys)
                self._y = np.searchsorted(self.classes_, ys).astype(np.int64)
            if not 2 <= len(self.classes_) <= MAX_CLASSES:
                raise ValueError(f"Servono da 2 a {MAX_CLASSES} classi (trovate {len(self.classes_)})")
        else:
            yv = _numeric_values(y)
            if not np.isfinite(yv).all():
                raise ValueError("Regressione: il target deve essere numerico")
            self._y_mean = yv.mean()
            var = yv.var()
            self._y_scale = 1.0 if _is_const(self._y_mean, var, len(yv)) else math.sqrt(var)
            self._y = yv
        return self

    def _run(self, X_query):
        q = as_columns(X_query, names=list(self._X))
        missing = [f for f in self._X if f not in q]
        if missing:
            raise ValueError(f"Colonne mancanti nella query: {missing}")
        q = {f: q[f] for f in self._X}
        n, nq, E = len(self._y), len(next(iter(q.values()))), self.n_estimators
        rng = np.random.default_rng(self.seed)

        # target per stimatore: shift delle classi / flip di segno (prima delle feature, come sdm)
        if self.task == "classification":
            K = len(self.classes_)
            offsets, perm = [], None
            for m in range(E):
                if m % K == 0:
                    perm = rng.permutation(K)
                offsets.append(int(perm[m % K]))
            ys = [(self._y - o) % K for o in offsets]
        else:
            signs = [-1.0 if rng.random() < 0.5 else 1.0 for _ in range(E)]
            yz = (self._y - self._y_mean) / self._y_scale
            ys = [(yz * s).astype(np.float32) for s in signs]

        feats = Features(self._X, q, n, self.categorical_features)
        if not feats.columns:
            raise ValueError("Tutte le feature sono costanti nel contesto")
        members = feats.build_members(E, rng)

        out_w = K if self.task == "classification" else 999
        result = np.empty((nq, out_w))
        for q0 in range(0, nq, self.chunk_size):
            q1 = min(nq, q0 + self.chunk_size)
            per_member = []
            for m, (x, mask) in enumerate(members):
                xb = np.concatenate([x[:n], x[n + q0:n + q1]])
                (logits,) = self.session.run(["logits"], {"x": xb, "y": ys[m], "categorical_mask": mask})
                logits = logits.astype(np.float64)
                if self.task == "classification":
                    per_member.append(logits[:, (np.arange(K) - offsets[m]) % K])  # annulla lo shift
                else:
                    per_member.append(np.sort(logits * signs[m] * self._y_scale + self._y_mean, axis=1))
            stack = np.stack(per_member)  # [E, chunk, out_w]
            if self.task == "classification":
                result[q0:q1] = stack.mean(axis=0)
            else:
                cut = int(0.2 * E)  # media troncata al 20%
                result[q0:q1] = np.sort(stack, axis=0)[cut:E - cut].mean(axis=0)
        return result

    def predict_proba(self, X_query):
        z = self._run(X_query)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, X_query, point="mean"):
        # Classificazione: classe più probabile. Regressione: media dei quantili (oppure "median").
        if self.task == "classification":
            return self.classes_[self.predict_proba(X_query).argmax(axis=1)]
        q = self._run(X_query)
        return q[:, 499] if point == "median" else q.mean(axis=1)

    def predict_quantiles(self, X_query, quantiles=(0.1, 0.5, 0.9)):
        # [n_query, len(quantiles)] dalla griglia q = 0.001 ... 0.999
        q = self._run(X_query)
        idx = np.clip(np.rint(np.asarray(quantiles) * 1000).astype(int) - 1, 0, 998)
        return q[:, idx]


class KumoClassifier(ClassifierMixin, BaseEstimator):
    # Stimatore sklearn, per usare Kumo come ultimo step di una Pipeline (e con cross_val_predict, GridSearchCV...).
    def __init__(self, n_estimators=8, seed=0, categorical_features=()):
        self.n_estimators = n_estimators
        self.seed = seed
        self.categorical_features = categorical_features

    def fit(self, X, y):
        # sklearn richiede fit(), ma non c'è training: si passa solo il contesto al modello
        self.model_ = KumoTabularONNX("classification", self.n_estimators, self.seed,
                                      self.categorical_features).set_context(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class KumoRegressor(RegressorMixin, BaseEstimator):
    # Come KumoClassifier, ma per la regressione; in più dà i quantili della distribuzione predetta.
    def __init__(self, n_estimators=8, seed=0, categorical_features=()):
        self.n_estimators = n_estimators
        self.seed = seed
        self.categorical_features = categorical_features

    def fit(self, X, y):
        # sklearn richiede fit(), ma non c'è training: si passa solo il contesto al modello
        self.model_ = KumoTabularONNX("regression", self.n_estimators, self.seed,
                                      self.categorical_features).set_context(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_quantiles(self, X, quantiles=(0.1, 0.5, 0.9)):
        return self.model_.predict_quantiles(X, quantiles)
