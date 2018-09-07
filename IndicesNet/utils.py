#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Alejandro Debus"
__email__ = "aledebus@gmail.com"

import torch
import pickle

def partitions(number, k):
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits = 3, subjects = 145, frames = 20):
    l = partitions(subjects, n_splits)
    fold_sizes = l * frames
    indices = np.arange(subjects * frames).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits = 3, subjects = 145, frames = 20):
    indices = np.arange(subjects * frames).astype(int)
    for test_idx in get_indices(n_splits, subjects, frames):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 80, 80)
    return x

def save_data(obj, fileobj):
    '''
    Store data (serialize)

    Arguments
        obj: object Python
        fileobj: 'my_obj.pickle'
    '''
    with open(fileobj, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
