#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers import initializers

class BLSTM_CRF(object):
    def __init__(self,hidden_unit,num_labels,num_layers=100,dropout_rate=0.5):
        self.hidden_nuit = hidden_unit
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels

    def _get_a_lstm_layer_(self):
        lstm_cell = rnn.LSTMCell(self.hidden_nuit)
        return lstm_cell

    def add_blstm_layers(self,inputs,lengths,is_training):
        #对输入dropout
        if is_training:
            inputs = tf.nn.dropout(inputs,self.dropout_rate)

        lstm_cell_fw = self._get_a_lstm_layer_()
        lstm_cell_bw = self._get_a_lstm_layer_()

        if(self.num_layers>1 ):
            lstm_cell_fw = rnn.MultiRNNCell([self._get_a_lstm_layer_(is_training) for _ in range(self.num_layers)],state_is_tuple=True)
            lstm_cell_bw = rnn.MultiRNNCell([self._get_a_lstm_layer_(is_training) for _ in range(self.num_layers)],state_is_tuple=True)


        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw,lstm_cell_bw,inputs,dtype=tf.float32)
        outputs = tf.concat(outputs,axis=2)

        #对输出dropout
        if is_training:
            outputs = tf.nn.dropout(outputs,self.dropout_rate)

        return outputs

    def project_layer(self,inputs):
        logits = tf.layers.dense(inputs,self.num_labels)
        return logits

    def add_crf_layer(self,inputs,labels,lengths):
        trans = tf.get_variable(
            "transitions",
            shape=[self.num_labels, self.num_labels],
            dtype=tf.float32)
        pred_ids, _ = tf.contrib.crf.crf_decode(inputs,trans,lengths)
        if(labels is not None):
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                inputs,labels,lengths,trans)
            loss = tf.reduce_mean(-log_likelihood)
            return loss,pred_ids,trans
        else:
            return None,pred_ids,trans

if __name__ == '__main__':
    pass