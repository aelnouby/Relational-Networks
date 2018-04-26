"""Visualization functions using Visdom
Author: Alaaeldin Ali
"""
from visdom import Visdom
import numpy as np
import torch


class VisdomPlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='relational_net'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, xlabel='iteration'):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel=xlabel,
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

    def draw(self, var_name, images):
        images = (images + 0.5) * 255
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.images(images, env=self.env)
        else:
            self.viz.images(images, env=self.env, win=self.plots[var_name])

    def print(self, var_name, text):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.text(text, env=self.env)
        else:
            self.viz.text(text, env=self.env, win=self.plots[var_name])



