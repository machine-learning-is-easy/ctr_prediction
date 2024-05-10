import numpy as np
import tensorflow as tf

from machine_learning.transformer.decoder import Decoder
from machine_learning.transformer.encoder import Encoder


class Transformer_CTR(tf.keras.Model):
    # define transformer ctr model
    def __init__(self, number_feature, sizeoffeature_list, num_layers, d_model, num_heads, dff, input_vocab_size,
                 pe_target_size, rate=0.1):
        super(Transformer_CTR, self).__init__()
        self.input_depth = input_vocab_size
        if number_feature != len(sizeoffeature_list):
            raise Exception("length of sizeoffeature_list is not equal to number_feature")
        self.num_encoder_feature = number_feature
        self.depth_feature = sizeoffeature_list
        self.encoder_feature = [tf.keras.layers.Embedding(sizeoffeature_list[_], d_model) for _ in range(number_feature)]

        self.encoder_sequence = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size=input_vocab_size, rate=rate)

        self.final_layer = tf.keras.layers.Dense(pe_target_size, activation="sigmoid")

    def __call__(self, inp_feature, inp_sequence):

        en_feature_output = []
        for _ in range(self.num_encoder_feature):
            indices = [_]
            feature_indices = tf.gather(inp_feature, indices, axis=1)
            input_feature_i = tf.one_hot(feature_indices,  depth=self.depth_feature[_])
            encode_embed = self.encoder_feature[_](input_feature_i)
            reshape_feature_output_i = tf.reshape(encode_embed, [-1, np.prod(encode_embed.shape[1:])])
            en_feature_output.append(reshape_feature_output_i)

        en_sequence_output = self.encoder_sequence(tf.convert_to_tensor(inp_sequence), training=True, mask=None)
        en_feature_output.append(tf.reshape(en_sequence_output, [-1, np.prod(en_sequence_output.shape[1:])]))
        concat_output = tf.concat(en_feature_output, axis=1)
        final_output = self.final_layer(tf.reshape(concat_output, [-1, np.prod(concat_output.shape[1:])]))# (batch_size, tar_seq_len, target_vocab_size)

        return final_output
