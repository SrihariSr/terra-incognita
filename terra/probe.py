"""Linear probes from hidden states to positions on the globe.

Why unit-sphere coordinates instead of latitude and longitude?
--------------------------------------------------------------
Longitude jumps from +180 to -180 at the dateline, so two places 10 km apart
(either side of the line, e.g. in Fiji) can have longitudes 359 degrees
apart. A linear probe trained on raw longitude is punished for that jump.
Instead every place becomes a point on a sphere of radius 1:

    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)

Nearby places now always have nearby (x, y, z). The probe predicts (x, y, z);
we push the prediction back onto the sphere by dividing by its length, then
convert back:

    lat = asin(z),   lon = atan2(y, x).

Great-circle distance
---------------------
The error of a prediction is the distance along the Earth's surface between
the true and the predicted point. For unit vectors a and b, the angle
between them is theta = atan2(|a x b|, a . b) (this form is accurate for
both tiny and large angles), and the distance is R * theta with R = 6,371 km.

Ridge regression
----------------
We fit Y ~ X W + b, where X holds standardised hidden states (each of the
3,584 features rescaled to mean 0 and standard deviation 1 on the training
set) and Y holds the (x, y, z) targets. Plain least squares would overfit:
there are more features than training places. Ridge adds a penalty
alpha * ||W||^2, which gives the closed form

    W = (X^T X + alpha I)^(-1) X^T Y          (primal form, d x d solve)
      = X^T (X X^T + alpha I)^(-1) Y          (dual form,  n x n solve)

The two are algebraically identical; we use whichever matrix is smaller.

Choosing alpha without touching the test set: leave-one-out (LOO) error has
a closed form for ridge. With hat matrix H = X (X^T X + alpha I)^(-1) X^T,
the LOO residual of row i is (y_i - yhat_i) / (1 - H_ii). Using the
eigen-decomposition of X^T X (or X X^T) we get H_ii for every alpha almost
for free, and pick the alpha with the lowest LOO error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0
ALPHAS = np.logspace(0, 6, 13)


# --------------------------------------------------------------- geometry

def latlon_to_xyz(lat, lon) -> np.ndarray:
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def xyz_to_latlon(xyz) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz / np.clip(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-12, None)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lat, lon


def great_circle_km(xyz_a, xyz_b) -> np.ndarray:
    a = np.asarray(xyz_a, dtype=np.float64)
    b = np.asarray(xyz_b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return EARTH_RADIUS_KM * np.arctan2(cross, dot)


# ------------------------------------------------------------------ ridge

@dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, w: np.ndarray | None = None) -> "Standardiser":
        if w is None:
            mu = X.mean(0)
            sd = X.std(0)
        else:
            w = w / w.sum()
            mu = (w[:, None] * X).sum(0)
            sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0))
        return cls(mu.astype(np.float64), np.where(sd < 1e-6, 1.0, sd).astype(np.float64))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def loo_alpha(Xs: np.ndarray, Y: np.ndarray, alphas=ALPHAS) -> tuple[float, dict]:
    """Pick alpha by closed-form leave-one-out error (see module docstring).

    Xs must be standardised and Y centred. Returns (best alpha, {alpha: mse}).
    """
    n, d = Xs.shape
    if n <= d:
        # Dual: eigen-decompose the n x n Gram matrix K = Xs Xs^T = U S U^T.
        K = Xs @ Xs.T
        s, U = np.linalg.eigh(K)
        UtY = U.T @ Y
        U2 = U ** 2
    else:
        # Primal: eigen-decompose Xs^T Xs = V S V^T; then Xs V plays the role of U sqrt(S).
        C = Xs.T @ Xs
        s, V = np.linalg.eigh(C)
        Z = Xs @ V                                  # n x d
        s_safe = np.where(s > 1e-9, s, 1e-9)
        U = Z / np.sqrt(s_safe)                     # columns orthonormal where s > 0
        UtY = U.T @ Y
        U2 = U ** 2
    s = np.clip(s, 0, None)
    scores = {}
    for a in alphas:
        shrink = s / (s + a)                        # how much each eigen-direction is kept
        Yhat = U @ (shrink[:, None] * UtY)
        h = U2 @ shrink                             # diagonal of the hat matrix
        loo = (Y - Yhat) / (1 - h)[:, None]
        scores[float(a)] = float(np.mean(loo ** 2))
    best = min(scores, key=scores.get)
    return best, scores


@dataclass
class Probe:
    """A fitted ridge probe: hidden state -> (x, y, z) on the unit sphere."""

    std: Standardiser
    W: np.ndarray        # d x 3, acting on standardised features
    b: np.ndarray        # 3
    alpha: float

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, alpha: float, w: np.ndarray | None = None) -> "Probe":
        """Fit with a fixed alpha. Optional per-row weights w (for the bootstrap)."""
        X = X.astype(np.float64)
        st = Standardiser.fit(X, w)
        Xs = st(X)
        if w is None:
            w = np.ones(len(X))
        b = (w[:, None] * Y).sum(0) / w.sum()
        Yc = Y - b
        n, d = Xs.shape
        if np.all(w == 1):
            Xw, Yw = Xs, Yc
        else:
            sw = np.sqrt(w)[:, None]
            Xw, Yw = Xs * sw, Yc * sw                # weighted least squares trick
        if n <= d:
            A = np.linalg.solve(Xw @ Xw.T + alpha * np.eye(n), Yw)
            W = Xw.T @ A
        else:
            W = np.linalg.solve(Xw.T @ Xw + alpha * np.eye(d), Xw.T @ Yw)
        return cls(st, W, b, alpha)

    def predict_xyz(self, X: np.ndarray) -> np.ndarray:
        return self.std(X.astype(np.float64)) @ self.W + self.b

    def predict_latlon(self, X: np.ndarray):
        return xyz_to_latlon(self.predict_xyz(X))

    def raw_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Express the probe on raw (unstandardised) hidden states: xyz = h @ A + c."""
        A = self.W / self.std.std[:, None]
        c = self.b - self.std.mean @ A
        return A, c


# ---------------------------------------------------------- evaluation

def folds_random(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row, uniformly at random."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def folds_grouped(groups, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row such that every member of a group shares a fold.

    Groups (countries or grid cells) are shuffled and then assigned greedily
    to the currently smallest fold, which keeps folds similar in size even
    when one group (e.g. the United States) is very large.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]  # big groups first
    sizes = np.zeros(k)
    fold_of = {}
    for gi in order:
        f = int(np.argmin(sizes))
        fold_of[uniq[gi]] = f
        sizes[f] += counts[gi]
    return np.array([fold_of[g] for g in groups])


def grid_cells(lat, lon, size_deg: float = 10.0) -> np.ndarray:
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    return (np.floor((lat + 90) / size_deg).astype(int) * 1000
            + np.floor((lon + 180) / size_deg).astype(int))


def cross_val_predict(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, alpha: float) -> np.ndarray:
    """Out-of-fold predictions: each row predicted by a probe that never saw it."""
    pred = np.zeros_like(Y, dtype=np.float64)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        p = Probe.fit(X[tr], Y[tr], alpha)
        pred[te] = p.predict_xyz(X[te])
    return pred


def error_summary(km: np.ndarray) -> dict:
    return {
        "median_km": float(np.median(km)),
        "mean_km": float(np.mean(km)),
        "p25_km": float(np.percentile(km, 25)),
        "p75_km": float(np.percentile(km, 75)),
        "within_500km": float(np.mean(km < 500)),
        "within_1000km": float(np.mean(km < 1000)),
        "n": int(len(km)),
    }


def baseline_error(Y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Error of the trivial 'predict the training-set mean direction' probe."""
    out = np.zeros(len(Y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = Y[tr].mean(0)
        out[te] = great_circle_km(np.broadcast_to(m, Y[te].shape), Y[te])
    return out
