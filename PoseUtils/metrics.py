"""
This page will contain more pose-related metrics as we go.
So far we have only the median translation and rotation errors.
"""

import numpy as np
import math
import transformations


def translations_from_trajectory(trajectory):
    """
    Returns the translations from a given sequence of transformations.
    """
    return np.array([tr[:3, -1] for tr in trajectory])


def quaternions_from_trajectory(trajectory):
    """
    Returns rotation quaternions from a sequence of transformation. 
    """
    return np.array(
        [transformations.quaternion_from_matrix(tr[:3, :3]) for tr in trajectory]
    )


def rotation_and_translation_error(
    true_trajectory: np.ndarray, pred_trajectory: np.ndarray
):
    """
    Calculates rotation and translation errors from two sequences of transformation matrics.
    The two sequences stand for the true and predicted trajectories.
    """
    translations_pred = translations_from_trajectory(pred_trajectory)
    translations_true = translations_from_trajectory(true_trajectory)
    rotations_true = quaternions_from_trajectory(true_trajectory)
    rotations_pred = quaternions_from_trajectory(pred_trajectory)

    results = []

    for j in range(translations_true.shape[0]):
        # TODO: Vectorize this benchmark!
        predicted_x = translations_pred[j]  # the last (layer) translation predictors
        predicted_q = rotations_pred[j]  # the last (layer) rotation predictor

        true_x = translations_true[j]
        true_q = rotations_true[j]

        # Compute Individual Sample Error
        q1 = true_q / np.linalg.norm(true_q)
        q2 = predicted_q / np.linalg.norm(predicted_q)
        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2 * np.arccos(d) * 180 / math.pi
        error_x = np.linalg.norm(true_x - predicted_x)
        results.append([error_x, theta])

    return results
