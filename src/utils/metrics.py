import numpy as np


def confusion_matrix_rl(successes):
    # Ground truth: reaching goal = positive
    y_true = np.ones(len(successes))  # Expected success
    y_pred = np.array(successes).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    FP = 0  # No false positives in RL success definition
    TN = 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }
