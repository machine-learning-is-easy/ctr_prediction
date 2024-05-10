# -*- coding: utf-8 -*-
import tensorflow as tf

def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """
    # sort by pred value, from small to big
    arr = sorted(raw_arr, key=lambda d:d[2])

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def _auc_arr(score):
  score_p = score[:,0]
  score_n = score[:,1]
  score_arr = []
  for s in score_p.tolist():
    score_arr.append([0, 1, s])
  for s in score_n.tolist():
    score_arr.append([1, 0, s])
  return score_arr

auc_metrics = tf.keras.metrics.AUC()

import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


def offline_evaluation(val_df, k):
    """
    :param val_df: evaluation dataframe includes y and yhat columns
    :param k: top k recommendations for calculating recall, precision and f1
    :return:
    """
    val_df['yhat_bin'] = val_df['yhat']\
        .apply(lambda x: np.array(x))\
        .apply(lambda x: x >= x[np.argsort(x)[-k:]].min())\
        .apply(lambda x: x.astype(int))

    val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2, 'auc'] = \
        val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2]\
            .apply(lambda x: roc_auc_score(x['y'], x['yhat']), axis=1)

    val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2, 'recall'] = \
        val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2] \
            .apply(lambda x: recall_score(x['y'], x['yhat_bin']), axis=1)

    val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2, 'precision'] = \
        val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2] \
            .apply(lambda x: precision_score(x['y'], x['yhat_bin']), axis=1)

    val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2, 'f1'] = \
        val_df.loc[val_df['y'].apply(lambda x: len(np.unique(x))) == 2] \
            .apply(lambda x: f1_score(x['y'], x['yhat_bin']), axis=1)

    print("AUC: ", val_df['auc'].mean(), "Top {}".format(k), "recall:", val_df['recall'].mean(),
          "precision:", val_df['precision'].mean(), "f1:", val_df['f1'].mean())

    return {
        'AUC': val_df['auc'].mean(),
        'Top {} recall'.format(k): val_df['recall'].mean(),
        'Top {} precision'.format(k): val_df['precision'].mean(),
        'Top {} f1'.format(k): val_df['f1'].mean(),
    }
