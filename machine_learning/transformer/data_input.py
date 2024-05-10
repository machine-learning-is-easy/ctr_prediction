# -*- coding: utf-8 -*-

class DataInput:
  def __init__(self, data, batch_size):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    if self.epoch_size * self.batch_size < len(self.data):
      self.epoch_size += 1
    self.i = 0

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    feature, seq, y, value = [], [], [], []
    for t in ts:
      feature.append(t[0])
      seq.append(t[1])
      y.append(t[2])
    return self.i, (feature, seq, y)
