import itertools
import warnings

from math import fsum
from collections import defaultdict

import numpy as np

from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support, f1_score, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")
"""
Ignore sklearn warnings:
*/sklearn/metrics/_classification.py:620:
RuntimeWarning: invalid value encountered in true_divide
k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)

and

*/sklearn/metrics/_classification.py:1221:
UndefinedMetricWarning: Precision is ill-defined and
being set to 0.0 in labels with no predicted samples.
Use `zero_division` parameter to control this behavior.
"""


def count_correct_2_to_5(y_pred, y_true_5_classes):
    """
    Map 2 (screening level) to 5 (DR level) classes and count
    correctly classified sample per DR level.

    Parameters
    ----------
    y_pred : array
        Predicted labels.
    y_true_5_classes : array
        5 classes true labels.

    Returns
    -------
    list
        List of str under the form:
        "class: correctly classified count/true class count (percentage of true class count)".
    """
    y_true_5_classes_count = defaultdict(lambda: 0)
    correctly_classified_count = y_true_5_classes_count.copy()
    for idx, pred in enumerate(y_pred):
        y_true_5_classes_count[y_true_5_classes[idx]] += 1
        if (pred[0] == 0. and y_true_5_classes[idx] in [0, 1]) or (pred[0] == 1. and y_true_5_classes[idx] in [2, 3, 4]):
            correctly_classified_count[y_true_5_classes[idx]] += 1

    print_correct = []
    for class_name in y_true_5_classes_count.keys():
        percent_correct = round(correctly_classified_count[class_name] / y_true_5_classes_count[class_name] * 100)
        print_info = f"{class_name}: {correctly_classified_count[class_name]}/{y_true_5_classes_count[class_name]} ({percent_correct}%)"
        print_correct.append(print_info)

    return sorted(print_correct)


def compute_scores(y_true, y_proba, mode, raw_labels, threshold=None, map_to_binary=False, map_to_5_classes=False, y_true_5_classes=None):
    """
    Compute confusion matrix, scores for each class including precision,
    recall, f1 score and support and global scores including f1 score macro
    and quadratic weighted cohen kappa score.

    Parameters
    ----------
    y_true : array
        True labels.
    y_proba : array
        Predicted probabilities.
    mode : str
        Mode in ['train', 'valid', 'test'].
    raw_labels : list
        List of labels.
    threshold : float or list of float.
        Threshold(s) to convert probablities into labels. Default None.
    map_to_binary : bool
        Optional, works only with len(raw_labels) > 2. Map to binary label before computing
        scores. Default False.
    map_to_5_classes : bool
        Optional, works only with len(raw_labels) == 2. Map to 5 classes label after computing
        scores to show the errors with regards to 5 classes labels. Default False.
    y_true_5_classes : array
        True 5 classes labels.
        Optional, needs to be set when map_to_5_classes is True. Default None.

    Returns
    -------
    scores : dict
        Dictionary containing the score: {"score_name": score}.
    """
    # Compute predicted labels from probabilities
    y_pred = compute_predicted_labels(y_proba, raw_labels, threshold)

    # Map to binary labels
    if len(raw_labels) > 2 and map_to_binary:
        raw_labels = [0., 1.]
        y_pred = (y_pred > 1).astype(float)
        y_true = (y_true > 1).astype(float)

    # Compute scores
    scores = {}
    # Per class scores
    scores[f"{mode}_confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=raw_labels)

    class_score_names = ["precision_per_class", "recall_per_class", "f1_per_class", "support_per_class"]
    # Note that in binary classification, recall of the positive class is also known as “sensitivity”;
    # recall of the negative class is “specificity”.
    class_scores = precision_recall_fscore_support(y_true, y_pred, labels=raw_labels)
    for idx, i in enumerate(class_scores):
        scores[f"{mode}_{class_score_names[idx]}"] = np.around(i, decimals=4)

    # Global scores
    # Unweighted mean of f1 per class
    scores[f"{mode}_f1_macro"] = round(f1_score(y_true, y_pred, average="macro"), 4)

    if len(raw_labels) == 2:
        # J statistic = sensitivity + specificity - 1
        scores[f"{mode}_j_statistic"] = round(scores[f"{mode}_recall_per_class"][1] + scores[f"{mode}_recall_per_class"][0] - 1, 4)

    # When num_class equal 2, weighted cohen kappa is the same as cohen kappa
    # The weighted kappa is calculated using a predefined table of weights
    # which measure the agreement between the two raters, the higher the
    # agreement the higher the weight and score.
    # k = (p_o - p_e)/(1 - p_e)
    # p_o = sum_i sum_j w_ij*p_ij -> observed agreement
    # p_e = sum_i sum_j w_ij*p_+i*p_+j -> expected agreement
    scores[f"{mode}_quadratic_weighted_kappa"] = round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 4)

    # Map correct to 5 classes
    if len(raw_labels) == 2 and map_to_5_classes:
        scores[f"{mode}_correctly_classified"] = count_correct_2_to_5(y_pred, y_true_5_classes)

    return scores


def compute_predicted_labels(y_proba, raw_labels, threshold=None):
    """
    Compute predicted labels.

    Parameters
    ----------
    y_proba : array
        Predicted probabilities.
    raw_labels : list
        List of labels.
    threshold : float or list of float.
        Threshold(s) to convert probablities into labels.

    Returns
    -------
    array
        Predicted labels.
    """
    if len(raw_labels) == 2:

        def pred(y_proba, threshold=None):
            if threshold is not None:
                return (y_proba >= threshold).astype(float)
            else:
                return y_proba.round()

    else:

        def pred(y_proba, threshold=None):
            if threshold is not None:
                # https://stats.stackexchange.com/questions/310952/how-the-probability-threshold-of-a-classifier-can-be-adjusted-in-case-of-multipl/310956#310956
                new_proba = y_proba * threshold[None, :]
                return np.argmax(new_proba, axis=1)
            else:
                return np.argmax(y_proba, axis=1)

    return pred(y_proba, threshold)


def find_best_threshold(y_true, y_proba, raw_labels, score_type='kappa'):
    """
    Find best threshold.

    Parameters
    ----------
    y_true : array
        True labels.
    y_proba : array
        Predicted probabilities.
    raw_labels : list
        List of labels.
    score_type : str
        Score of interest for best threshold.

    Returns
    -------
    float or list of float
        Threshold(s) to convert probablities into labels.
    """
    if len(raw_labels) == 2:
        if score_type == 'kappa' or score_type == 'f1_macro':
            scores = []
            threshold = np.arange(0, 1, 0.001)
            for thresh in threshold:
                y_pred = compute_predicted_labels(y_proba, raw_labels, thresh)
                if score_type == 'f1_macro':
                    scores.append(f1_score(y_true, y_pred, average="macro"))
                else:
                    scores.append(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
            return threshold[np.nanargmax(np.array(scores))]
        elif score_type == 'J_statistic':
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            J = tpr - fpr
            return thresholds[np.argmax(J)]
        elif score_type == 'f1_c1':
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            fscore = (2 * precision * recall) / (precision + recall)
            return thresholds[np.argmax(fscore)]

    else:
        scores = []
        possible_priors = np.arange(0.1, 0.7, 0.1)
        threshold = [i for i in itertools.product(possible_priors, repeat=5) if fsum(i) == 1]
        for thresh in threshold:
            y_pred = compute_predicted_labels(y_proba, raw_labels, np.array(thresh))
            if score_type == 'kappa':
                scores.append(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
            elif score_type == 'f1_macro':
                scores.append(f1_score(y_true, y_pred, average="macro"))
        return np.array(threshold[np.nanargmax(np.array(scores))])
