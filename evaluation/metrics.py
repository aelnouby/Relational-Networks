"""Helper functions for measuring accuracy, precision, ....
Author: Alaaeldin El-Nouby"""

import numpy as np
from helpers.visualize import VisdomPlotter


def get_accuracy(predictions, target):
    """
    Measure accuracy
    :param predictions: numpy array of logits
    :param target: numpy array of class indices
    :return: accuracy: float type
    """
    if type(predictions) is not np.ndarray:
        raise Exception('Predictions must be numpy array')

    if type(target) is not np.ndarray:
        raise Exception('target must be numpy array')

    length = len(predictions)

    true_positives = 0
    for i in range(length):
        if np.argmax(predictions[i]) == target[i]:
            true_positives += 1

    accuracy = true_positives / length

    return accuracy


class MetricsInfo(object):
    """An object that keeps track of accuracies over each epoch
    then  plot, clears the cache and starts tracking again"""
    def __init__(self, env_name):
        self.accuracies = []
        self.viz = VisdomPlotter(env_name=env_name)

    def track_accuracy(self, predictions, target):
        """Measure accuracies of (total, fertile, infertile) and adds to a cache until plot is called"""
        accuracy = get_accuracy(predictions.data.cpu().numpy(), target.data.cpu().numpy())
        self.accuracies.append(accuracy)

    def plot_accuracies(self, epoch, train=True):
        """Plots the accuracies currently in cache using Visdom, then clears cache
        :param: epoch
        """
        split = 'train' if train else 'valid'

        self.viz.plot('accuracy', split, epoch, np.array(self.accuracies).mean())

        self.accuracies = []

