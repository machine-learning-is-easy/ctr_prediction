# -*- coding: utf-8 -*-
import os
import pickle
import json

from utilities.GoogleOperation import GoogleCloudPersistor as GoogleOperation
from machine_learning.transformer.transformerctr import Model
from settings import get_settings
from machine_learning.transformer.training_dataset import prepare_training_data

num_layers = 2
d_model = 64
num_heads = 8
dff = 128
input_vocab_size = 8500
target_vocab_size = 8000
pe_input = 10000
pe_target = 6000

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

def model_training(content_space_id, training=True):

    if get_settings().remote:
        training_data = prepare_training_data(content_space_id)
        
        train_set = training_data.train_set
        dev_set = training_data.dev_set
        item2id = training_data.item2id
        os2id = training_data.os2id
        pc2id = training_data.pc2id
        item_count = training_data.item_count
        agent_count = training_data.agent_count
        len_week = training_data.len_week
        len_hour = training_data.len_hour
        test_set = training_data.test_set
    else:
        with open(os.path.join(CURRENT_FOLDER, '../raw_data/{}/dataset_trainingtesting.pkl'.
                format(content_space_id)), 'rb') as f:
            train_set = pickle.load(f)
            dev_set = pickle.load(f)
            item2id, os2id, pc2id = pickle.load(f)
            # item_count, agent_count, len_week, len_hour, len_postcode = pickle.load(f)
            item_count, agent_count, len_week, len_hour = pickle.load(f)
            test_set = pickle.load(f)

    if len(train_set) > 0 and len(test_set) > 0 and len(dev_set) > 0:
        ctrmodel = Model()
        ctrmodel.create_model(num_feature=3, sizeoffeature_list=[agent_count, len_week, len_hour],
                         input_sequence_size=item_count, num_layers=num_layers, d_model=d_model,
                         num_heads=num_heads, dff=dff, input_vocab_size=item_count, target_vocab_size=item_count,
                         dropout_rate=0.1,
                         content_space_id=content_space_id)
        if training:
            ctrmodel.training_epoch(train_dataset=train_set, d_model=d_model)
            ctrmodel.restore(d_model=d_model)
            if get_settings().remote:                
                export_path = os.path.join(
                    "gs://",
                    get_settings().neural_recommender_bucket,
                    get_settings().neural_model_gcp_folder,
                    content_space_id,
                    "saved_model"
                )
                ctrmodel.export(export_path)                
            else:
                ctrmodel.export(os.path.join(CURRENT_FOLDER, "checkpoints/{}/saved_model".format(content_space_id)))

                with open(os.path.join(CURRENT_FOLDER, "checkpoints/{}/dictionary.json".format(content_space_id)), 'w') as f:
                    f.write(json.dumps({"item2id": item2id, "os2id":os2id}))
        else:
            ctrmodel.restore(d_model=d_model)
        
        test_result = ctrmodel._eval(dataset=test_set)
        print(test_result)

        if get_settings().remote:
            google_persistor = GoogleOperation(
                project=get_settings().project_id,
                bucket=get_settings().neural_recommender_bucket
            )
            google_persistor.set_google_cloud()

            # Save dictionary
            item_dict = { "item2id": item2id, "os2id":os2id }
            dict_path = os.path.join(get_settings().neural_model_gcp_folder, content_space_id, "dictionary.json")
            google_persistor.upload_str_googlecloud(json.dumps(item_dict), dict_path)

            # Save evaluation
            eval_path = os.path.join(get_settings().neural_model_gcp_folder, content_space_id, "evaluation.json")
            google_persistor.upload_str_googlecloud(json.dumps(test_result), eval_path)

            # Delete checkpoints after the training is complete to avoid confusion for the next training workflow
            checkpoints_path = os.path.join(get_settings().neural_model_gcp_folder, content_space_id, "checkpoints")
            google_persistor.delete_folder(checkpoints_path)
        else:
            #UPLOAD MODEL FILES
            destination_folder = os.path.join(get_settings().neural_model_gcp_folder, content_space_id)
            credential_file = os.path.join(CURRENT_FOLDER, "../../credentials/OpenHouse-AI Prod-bf7b3a94416b.json")
            google_persistor = GoogleOperation(bucket=get_settings().neural_recommender_bucket,
                                            confidential_file=credential_file)
            google_persistor.set_google_cloud()

            ctrmodel.uploadmodel(google_persistor, destination_folder)

        return test_result
