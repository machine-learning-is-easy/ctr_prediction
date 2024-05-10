# -*- coding: utf-8 -*-
import pandas as pd
from machine_learning.plot_function import evaluation_plot
from machine_learning.transformer.downloaddata import download_data
from machine_learning.transformer.training import model_training
from machine_learning.transformer.training_dataset import prepare_training_data
from settings import get_settings


training = True
evaluation_data = pd.DataFrame()
for id in get_settings().content_space_id:
    if training:
        download_data(content_space_id=id)
        prepare_training_data(content_space_id=id)
    result = model_training(content_space_id=id, training=training)
    if result is not None:
        result = {key: [round(value, 2)] for key, value in result.items()}
        result["content_spacy_id"] = id
        evaluation_i = pd.DataFrame(result)
        evaluation_data = evaluation_data.append(evaluation_i)

evaluation_plot(evaluation_data)
