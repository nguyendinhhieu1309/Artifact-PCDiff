"""Evaluation metrics defined by the Artifact-PCDiff paper.

All inputs must use the same coordinate frame and normalization. The distance
factors compare a predicted repair against repeated 3,072-point samples from a
dense ground-truth repair, as specified in the paper.
"""

import numpy as np
from scipy.spatial import cKDTree


def _as_point_set(points, name):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("{} must be a non-empty [N, 3] point array".format(name))
    if not np.isfinite(points).all():
        raise ValueError("{} contains non-finite coordinates".format(name))
    return points


def chamfer_distance_squared(points_a, points_b):
    """Return symmetric squared Chamfer Distance, with an unsquared mean per side."""
    points_a = _as_point_set(points_a, "points_a")
    points_b = _as_point_set(points_b, "points_b")
    distances_a = cKDTree(points_b).query(points_a, k=1)[0]
    distances_b = cKDTree(points_a).query(points_b, k=1)[0]
    return float(np.mean(np.square(distances_a)) + np.mean(np.square(distances_b)))


def hausdorff_distance(points_a, points_b):
    """Return symmetric Hausdorff Distance using Euclidean nearest-neighbor distances."""
    points_a = _as_point_set(points_a, "points_a")
    points_b = _as_point_set(points_b, "points_b")
    distances_a = cKDTree(points_b).query(points_a, k=1)[0]
    distances_b = cKDTree(points_a).query(points_b, k=1)[0]
    return float(max(np.max(distances_a), np.max(distances_b)))


def _interface_band(repair, fragment, count):
    repair = _as_point_set(repair, "repair")
    fragment = _as_point_set(fragment, "fragment")
    count = min(int(count), len(repair))
    if count <= 0:
        raise ValueError("interface point count must be positive")
    distances = cKDTree(fragment).query(repair, k=1)[0]
    indices = np.argpartition(distances, count - 1)[:count]
    return repair[indices]


def boundary_chamfer_distance(predicted_repair, target_repair, fragment, interface_points=64):
    """Compute paper BCD on repair-side bands nearest to the observed fragment."""
    predicted_band = _interface_band(predicted_repair, fragment, interface_points)
    target_band = _interface_band(target_repair, fragment, interface_points)
    return chamfer_distance_squared(predicted_band, target_band)


def sampling_normalized_factors(
    predicted_repair,
    dense_target_repair,
    evaluation_points=3072,
    trials=20,
    epsilon=1e-8,
    seed=42,
):
    """Compute paper CDF/HDF factors using independent dense-GT resamples.

    For every trial, two independent ``evaluation_points`` samples estimate the
    finite-sampling baseline. The first is also compared with the prediction.
    """
    predicted_repair = _as_point_set(predicted_repair, "predicted_repair")
    dense_target_repair = _as_point_set(dense_target_repair, "dense_target_repair")
    evaluation_points = int(evaluation_points)
    trials = int(trials)
    if evaluation_points <= 0 or trials <= 0:
        raise ValueError("evaluation_points and trials must be positive")
    if len(predicted_repair) != evaluation_points:
        raise ValueError(
            "predicted_repair must contain {} points; got {}".format(
                evaluation_points, len(predicted_repair)
            )
        )
    if len(dense_target_repair) < evaluation_points:
        raise ValueError(
            "dense_target_repair must contain at least {} points; got {}".format(
                evaluation_points, len(dense_target_repair)
            )
        )

    rng = np.random.default_rng(seed)
    predicted_cd = []
    baseline_cd = []
    predicted_hd = []
    baseline_hd = []
    for _ in range(trials):
        first = dense_target_repair[
            rng.choice(len(dense_target_repair), size=evaluation_points, replace=False)
        ]
        second = dense_target_repair[
            rng.choice(len(dense_target_repair), size=evaluation_points, replace=False)
        ]
        predicted_cd.append(chamfer_distance_squared(predicted_repair, first))
        baseline_cd.append(chamfer_distance_squared(first, second))
        predicted_hd.append(hausdorff_distance(predicted_repair, first))
        baseline_hd.append(hausdorff_distance(first, second))

    mean_baseline_cd = float(np.mean(baseline_cd))
    mean_baseline_hd = float(np.mean(baseline_hd))
    return {
        "CDF": float(np.mean(predicted_cd) / (mean_baseline_cd + epsilon)),
        "HDF": float(np.mean(predicted_hd) / (mean_baseline_hd + epsilon)),
        "CD_sampling_baseline": mean_baseline_cd,
        "HD_sampling_baseline": mean_baseline_hd,
    }


def summarize_metric(values):
    """Return the per-object summary statistics reported in the paper."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite one-dimensional array")
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }
