#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf

from model.blstm_crf_model import BLSTM_CRF


def model_fn(features,labels,mode,params):
    inputs,lengths = features

    #对输入dropout
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # inputs = tf.layers.dropout(inputs, rate=params['dropout_rate'], training=is_training)

    # LSTM
    print('构造LSTM层')

    # 转换为lstm时间序列输入
    t = tf.transpose(inputs, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['hidden_units'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)

    print('LSTM联合层')
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=lengths)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=lengths)  # shape 49*batch_size*100
    outputs = tf.concat([output_fw, output_bw], axis=-1)  # 40*batch_size*200
    # 转换回正常输出
    outputs = tf.transpose(outputs, perm=[1, 0, 2])  # batch_size*40*200
    #对输出dropout
    print('dropout')
    outputs = tf.layers.dropout(outputs, rate=params['dropout_rate'], training=is_training)

    # 全连接层
    logits = tf.layers.dense(outputs, params['num_labels'])  # batch_size*40*len(tags)

    logits = tf.nn.softmax(logits)

    print('CRF层')
    crf_params = tf.get_variable("transitions", [params['num_labels'], params['num_labels']], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, lengths)



    if(mode == tf.estimator.ModeKeys.PREDICT):
        print('预测')
        # if(params['ilp']):
        #     predictions = {
        #         'logits':logits,
        #     }
        # else:
        predictions = {
            'pred_ids': pred_ids
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)
    else:
        # Loss
        print('loss计算')
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, labels, lengths, crf_params)
        loss = tf.reduce_mean(-log_likelihood)
        if(mode == tf.estimator.ModeKeys.EVAL):
            print('评估')
            weights = tf.sequence_mask(lengths, maxlen=params['max_sequence_length'])
            metrics = {
                'acc': tf.metrics.accuracy(labels, pred_ids, weights)
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])

            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        else:
            print('训练')
            train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
if __name__ == '__main__':
    pass