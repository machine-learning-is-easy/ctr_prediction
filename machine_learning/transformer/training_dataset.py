import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from utilities.bigquery import get_neural_corpus_dataframe
from machine_learning.transformer.utilities import Training_Data
from settings import get_settings

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
TRAINING_FROM = 7*30
TRAINING_TO = 1*30
TESTING_FROM = 1*30
TESTING_TO = 0

def build_map(df, col_name):
  key = sorted([str(k) for k in df[col_name].unique().tolist()])
  m = dict(zip(key, range(len(key))))
  return m, key

def prepare_training_data(content_space_id):
    if get_settings().remote:
        training_data_set = get_neural_corpus_dataframe(content_space_id, TRAINING_FROM, TRAINING_TO)
        test_set = get_neural_corpus_dataframe(content_space_id, TESTING_FROM, TESTING_TO)
    else:
        training_file_dir = os.path.join(CURRENT_FOLDER, "../../machine_learning/raw_data/{}/training.csv".format(content_space_id))
        test_file_dir = os.path.join(CURRENT_FOLDER, "../../machine_learning/raw_data/{}/testing.csv".format(content_space_id))        
        training_data_set = pd.read_csv(training_file_dir)
        test_set = pd.read_csv(test_file_dir)
    
    all_data_set = training_data_set.append(test_set)
    # item to id
    item2id, item_key = build_map(all_data_set, 'id')
    item_count = len(item2id)
    # version to id
    os2id, version_key = build_map(all_data_set, 'os')
    pc2id, pc_key = build_map(all_data_set, 'postcode')
    agent_count = len(os2id)

    training_data_set["id"] = training_data_set["id"].map(lambda x: item2id[str(x)])
    training_data_set["os"] = training_data_set["os"].map(lambda x: os2id[str(x)])
    training_data_set["postcode"] = training_data_set["postcode"].map(lambda x: pc2id[str(x)])

    test_set["id"] = test_set["id"].map(lambda x: item2id[str(x)])
    test_set["os"] = test_set["os"].map(lambda x: os2id[str(x)])
    test_set["postcode"] = test_set["postcode"].map(lambda x: pc2id[str(x)])

    len_week = 7
    len_hour = 24

    Max_len = 10
    def generating_sequence(data_set):
        train_set = []
        for categories, hist in data_set.groupby(['user_id', "session_id"]):
            pos_list = hist['id'].tolist()
            agent_i = hist["os"].tolist()[0]
            weekday = hist["day_of_week"].tolist()[0]
            hourofday = hist["hour_of_the_day"].tolist()[0]

            for i in range(1, len(pos_list)):
                sub_hist = pos_list[:i]
                if len(sub_hist) >= Max_len:
                    sub_hist = sub_hist[-Max_len:]
                else:
                    sub_hist = [item_count] * (Max_len - len(sub_hist)) + sub_hist
                if len(sub_hist) != Max_len:
                    a = 1
                assert len(sub_hist) == Max_len

                output = [1 if item in pos_list else 0 for item in range(item_count + 1)]

                train_set.append(
                    [[agent_i, weekday, hourofday], sub_hist, output])

        return train_set

    # create training data
    train_set = generating_sequence(training_data_set)
    train_set, dev_set = train_test_split(train_set, test_size=0.2, random_state=42, shuffle=True)

    #create test data
    test_set = generating_sequence(test_set)

    if get_settings().remote:
        return Training_Data(train_set, dev_set, item2id, os2id, pc2id, item_count + 1, agent_count, len_week, len_hour, test_set)        
    else:
        with open(os.path.join(CURRENT_FOLDER, '../raw_data/{}/dataset_trainingtesting.pkl'.format(content_space_id)),
                'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dev_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((item2id, os2id, pc2id), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((item_count + 1, agent_count, len_week, len_hour), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set,f, pickle.HIGHEST_PROTOCOL)
