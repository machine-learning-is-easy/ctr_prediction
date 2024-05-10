# -*- coding: utf-8 -*-
import os

from machine_learning.transformer.transformerctr import Model


def download_data(content_space_id):
    CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
    training_file_dir = os.path.join(CURRENT_FOLDER, "../../machine_learning/raw_data/{}/training.csv".
                                     format(content_space_id))
    test_file_dir = os.path.join(CURRENT_FOLDER, "../../machine_learning/raw_data/{}/testing.csv".
                                 format(content_space_id))
    training_data_end = "2020-09-01"
    training_data_begin = "2020-01-01"
    test_data_begin = "2020-09-01"
    test_data_end = "2020-10-15"
    Model.download(file_dir=training_file_dir, date_begin=training_data_begin, date_end=training_data_end,
                   content_space_id=content_space_id)
    Model.download(file_dir=test_file_dir, date_begin=test_data_begin, date_end=test_data_end,
                   content_space_id=content_space_id)
