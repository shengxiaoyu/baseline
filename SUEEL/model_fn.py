#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
from model.blstm_crf_model import BLSTM_CRF
from SUEEL.ilp_solver import optimize
from SUEEL.config import CRF_TRANS

def model_fn(features,labels,mode,params):
    tf.enable_eager_execution()
    inputs,lengths = features
    model = BLSTM_CRF(params['hidden_units'],params['num_labels'],params['num_layers'],params['dropout_rate'])
    if(mode==tf.estimator.ModeKeys.TRAIN):
        outputs = model.add_blstm_layers(inputs,lengths,True)
    else:
        outputs = model.add_blstm_layers(inputs,lengths,False)

    logits = model.project_layer(outputs)
    # 将logits规整化到0~1
    logits = tf.nn.softmax(logits)


    loss,pred_ids,trans = model.add_crf_layer(logits,labels,lengths)



    if(mode == tf.estimator.ModeKeys.PREDICT):
        print('预测')
        if(params['ilp']):
            predictions = {
                'logits':logits,
            }
        else:
            predictions = {
                'pred_ids': pred_ids
            }
        #使用ilp-solver获取最优解：

        return tf.estimator.EstimatorSpec(mode,predictions=predictions)
    elif(mode == tf.estimator.ModeKeys.EVAL):
        print('评估')
        weights = tf.sequence_mask(lengths, maxlen=params['max_sequence_length'])
        # indices = [item[1] for item in CONFIG.TAG_2_ID.items() if (item[0] != '<pad>' and item[0] != 'O')]
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