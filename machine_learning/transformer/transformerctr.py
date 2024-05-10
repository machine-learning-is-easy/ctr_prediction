import os
import pickle
import shutil
from pathlib import Path
import json

import matplotlib as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from settings import get_settings

from datasource.get_feature_sequence import retrieve_per_session_data_flattened
from machine_learning.transformer.data_input import DataInput
from machine_learning.transformer.metrics import offline_evaluation
from machine_learning.transformer.optimizer import CustomSchedule
from machine_learning.transformer.transformer import Transformer_CTR
from machine_learning.transformer.utilities import create_padding_mask, create_look_ahead_mask

EPOCHS = 20

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

loss_object = tf.keras.losses.BinaryCrossentropy(label_smoothing=True, from_logits=True)

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

class Model(object):
    def __init__(self):
        pass

    def create_model(self,  num_feature, sizeoffeature_list, input_sequence_size, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, dropout_rate, content_space_id):

        inputs_feature = tf.keras.Input(shape=(num_feature,), name="feature_inputs", dtype=tf.dtypes.int32)
        inputs_sequence = tf.keras.Input(shape=(10,), name="sequence_inputs", dtype=tf.dtypes.int32)
        attention_model = Transformer_CTR(num_feature, sizeoffeature_list,
                                     num_layers=num_layers, d_model=d_model,
                                     num_heads=num_heads, dff=dff,
                                     input_vocab_size=input_vocab_size,
                                     pe_target_size=target_vocab_size,
                                     rate=dropout_rate)
        probability = attention_model(inp_feature=inputs_feature, inp_sequence=inputs_sequence)
        self.model = tf.keras.Model(inputs=[inputs_feature, inputs_sequence], outputs=probability)

        self.target_vocab_size = target_vocab_size
        self.content_space_id = content_space_id
        self.train_loss = tf.keras.losses.BinaryCrossentropy(name='train_loss')
        self.train_accuracy = tf.keras.metrics.BinaryCrossentropy(name='train_accuracy')
        self.auc_metrics = tf.keras.metrics.AUC()

    def define_optimizer(self, model):
        learning_rate = CustomSchedule(model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.train_loss(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_check_point(self):
        if get_settings().remote:            
            checkpoint_path = os.path.join(
                "gs://",
                get_settings().neural_recommender_bucket,
                get_settings().neural_model_gcp_folder,
                self.content_space_id,
                "checkpoints"
            )
        else:    
            checkpoint_path = os.path.join(CURRENT_FOLDER, "checkpoints/{}/train".format(self.content_space_id))

        ckpt = tf.train.Checkpoint(transformer=self.model,
                                   optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def download(file_dir, content_space_id, date_begin=None, date_end=None):
        try:
            retrieve_per_session_data_flattened(csv_file_path=file_dir, content_space_id=content_space_id,
                                                date_str1=date_begin, date_str2=date_end)
        except Exception:
            raise Exception("Failed to download files")

    def train_step(self, feature, seq, tar_list,):
        tar_inp = tar_list

        with tf.GradientTape()    as tape:
            probability = self.model({
                "feature_inputs": np.array(feature, dtype=np.int32),
                "sequence_inputs": np.array(seq, dtype=np.int32)})
            self.loss = self.loss_function(tar_inp, probability)

        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.auc_metrics(tar_inp, probability)

    def _eval(self, dataset):
        self.auc_metrics.reset_states()
        batch_num = 0
        loss = 0
        total_auc = 0
        y_hat = []
        y = []
        for (batch, (feature, seq, tar)) in DataInput(dataset, 32):
            with tf.GradientTape() as tape:
                probability = self.model({
                    "feature_inputs": np.array(feature, dtype=np.int32),
                    "sequence_inputs": np.array(seq, dtype=np.int32)})
                y_hat += probability.numpy().tolist()
                y += tar

                self.loss = self.loss_function(tar, probability)
                loss += self.loss
                batch_num += 1
                total_auc += tf.keras.metrics.AUC()(tar, probability)
        if batch_num > 0:
            print('{} Evaluation Loss {:.4f} evaluation AUC {:.4f}'.format(self.content_space_id, loss, total_auc/batch_num))
        else:
            print('{} Evaluation Loss {:.4f} evaluation AUC {:.4f}'.format(self.content_space_id, loss, total_auc))
        return offline_evaluation(pd.DataFrame({"yhat":y_hat, "y":y}), 5)

    # @tf.function(input_signature=train_step_signature)
    def training_epoch(self, train_dataset, d_model):
        self.define_optimizer(d_model)
        self.create_check_point()

        for epoch in range(EPOCHS):
            self.auc_metrics.reset_states()
            for (batch, (feature, seq, tar_list)) in DataInput(train_dataset, 32):
                self.train_step(feature=feature, seq=seq, tar_list=tar_list)
                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} AUC {:.4f}'.format(
                        epoch + 1, batch, self.loss,
                        self.auc_metrics.result()))
            if (epoch + 1) % 1 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))

    def plot_attention_weights(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer_pt.encode(sentence)
        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')
            fontdict = {'fontsize': 10}
            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))
            ax.set_ylim(len(result) - 1.5, -0.5)
            ax.set_xticklabels(
                ['<start>'] + [self.tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer_en.decode([i]) for i in result
                                if i < self.tokenizer_en.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

    def evaluate(self, inp_feature, inp_sentence):
        start_token = [self.tokenizer_pt.vocab_size]
        end_token = [self.tokenizer_pt.vocab_size + 1]

        inp_sentence = start_token + self.tokenizer_pt.encode(inp_sentence) + end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        # as the target is english, the first word to the transformer should be the
        # english start token.
        decoder_input = [self.tokenizer_en.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.MAX_LENGTH):
            enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(
                encoder_input, output)

            predictions, attention_weights = self.transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.tokenizer_en.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def export(self, path):
        if get_settings().remote:
            self.model.save(path)            
        else:
            tf.keras.models.save_model(model=self.model, filepath=path)

    def restore(self, d_model):
        try:
            self.define_optimizer(d_model)
            self.create_check_point()
        except:
            raise Exception("Failed to load model")

    def suggest(self, feature, seq):
        # history is a dictionary,
        # history is a list of keys
        # feature is a dict of fearute: os, dayOfWeek, hourOfDay
        # return a list of keys reverse order by the probability
        def patch_seq(seq_ids):
            if len(seq_ids) >= 10:
                sub_hist = seq_ids[-10:]
            else:
                sub_hist = [len(self.item2id)] * (10 - len(seq_ids)) + seq_ids
            return sub_hist

        seq_ids = [self.item2id[key] if key in self.item2id else len(self.item2id) for key in seq]
        patched_seq = patch_seq(seq_ids)
        feature_ids = []

        if 'os' in feature and feature['os'] in self.os2id:
            feature_ids.append(self.os2id[feature["os"]])
        else:
            feature_ids.append(0)

        feature_ids.append(feature['dayOfWeek'])
        feature_ids.append(feature['hourOfDay'])
        with tf.GradientTape() as tape:
            probability = self.model([np.array([feature_ids], dtype=np.int32), np.array([patched_seq], dtype=np.int32)])
            self.id2item = {value: key for key, value in self.item2id.items()}
            
            item_probs = {self.id2item[index]: value for index, value in enumerate(probability.numpy().tolist()[0])
                    if index in self.id2item.keys()}
            
            return [e[0] for e in sorted(item_probs.items(), key=lambda item: item[1], reverse=True)]

    def load(self, model_dir, google_persistor, blobfolder):
        # download files to local directory and load model
        self.downloadmodel(google_persistor, blobfolder, model_dir)
        self.load_local_model(model_dir)
        self.load_dictionary(model_dir)

    def load_dictionary(self, model_dir):
        with open(os.path.join(model_dir, "dictionary.json"), 'r') as f:
            train_set = json.load(f)
        self.item2id = train_set['item2id']
        self.os2id = train_set['os2id']

    def load_local_model(self, checkpoint_dir):
        latest = os.path.join(checkpoint_dir, "saved_model")
        self.model = tf.saved_model.load(latest)
        print("Successfully load autocomplete model")

    def uploadmodel(self, google_persistor, gcp_model_directory):
        # blob is object of google bucket
        # upload dictionary.json and model files to gcp bucket
        # google_persistor: gcp bucket operation instance
        # destination_folder: gcp bucket path
        destination_folder = os.path.join(gcp_model_directory, self.content_space_id)
        model_local_direct = os.path.join(CURRENT_FOLDER, 'checkpoints/{}'.format(self.content_space_id))
        dictionary_file = os.path.join(model_local_direct, 'dictionary.json')
        # upload dictionary file
        google_persistor.upload_blob(destination_blob_name=os.path.join(destination_folder, "dictionary.json"),
                                     source_file_name=dictionary_file)

        saved_model_file = os.path.join(model_local_direct, "saved_model", "saved_model.pb")
        google_persistor.upload_blob(source_file_name=saved_model_file,
                                     destination_blob_name=os.path.join(destination_folder, 'saved_model',
                                                                        'saved_model.pb'))

        variables_files = [x for x in Path(os.path.join(model_local_direct,"saved_model",
                                                        "variables")).iterdir() if 'temp' not in str(x)]
        for file in variables_files:
            google_persistor.upload_blob(source_file_name=str(file), destination_blob_name=
            os.path.join(destination_folder, "saved_model", "variables", str(file).split("/")[-1]))

    def downloadmodel(self, google_persistor, blobfolder, destination_folder):
        # find the most recent files in blobpath
        blob_vob_file = os.path.join(blobfolder, 'dictionary.json')
        destination_dictionary_file_path = os.path.join(destination_folder, "dictionary.json")

        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)
        else:
            shutil.rmtree(destination_folder)
            os.makedirs(destination_folder)
        google_persistor.download_blob(blob_vob_file, destination_dictionary_file_path)

        destination_model_folder = os.path.join(destination_folder, "saved_model")

        if not os.path.isdir(destination_model_folder):
            os.makedirs(destination_model_folder)
        else:
            shutil.rmtree(destination_model_folder)
            os.makedirs(destination_model_folder)

        blob_saved_model_file = os.path.join(blobfolder, "saved_model", "saved_model.pb")
        destination_model_file = os.path.join(destination_model_folder, "saved_model.pb")
        google_persistor.download_blob(blob_saved_model_file, destination_model_file)

        variables_file = os.path.join(blobfolder, "saved_model", "variables")
        # destination_variable_file_name = os.path.join(destination_model_file, "variables")
        bloblist = google_persistor.blobs_list()
        varibles_files = [file.name for file in bloblist if variables_file in file.name and not file.name.endswith('/')]

        for blobfile in varibles_files:
            destination_variable_folder_name = os.path.join(destination_model_folder, "variables")
            if not os.path.isdir(destination_variable_folder_name):
                os.makedirs(destination_variable_folder_name)

            destination_variable_file_name = os.path.join(destination_variable_folder_name,
                                                          blobfile.split("/")[-1])
            google_persistor.download_blob(blobfile, destination_variable_file_name)
