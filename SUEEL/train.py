#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
import os
from sklearn_crfsuite.metrics import flat_classification_report
from SUEEL.model_fn import model_fn
from SUEEL.input_fn import input_fn
from SUEEL.input_fn import generator_fn
from SUEEL.config import getArgs
import functools
from SUEEL.ilp_solver import optimize

def main(args):
    print(args)

    tf.enable_eager_execution()

    # 配置哪块gpu可见
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    output_dir = os.path.join(args.root_dir, 'output_' + str(args.num_epochs) + '_' + str(
        args.batch_size)+'_'+str(args.num_layers) + '_' + 'baseline_SUEEL')
    if args.isTraining:
        if os.path.exists(output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                print('清除历史训练记录')
                del_file(output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    # check output dir exists
    if not os.path.exists(output_dir):
        print('创建output文件夹')
        os.mkdir(output_dir)

    session_config = tf.ConfigProto(
        # 是否打印使用设备的记录
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        # 是否允许自行使用合适的设备
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=120,
        session_config=session_config
    )
    params = {
        'hidden_units': args.hidden_units,
        'num_layers': args.num_layers,
        'max_sequence_length': args.max_sequence_length,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'num_labels':args.num_labels,
        'id2tag':args.id2tag,
        'trigger_ids':args.trigger_ids,
        'trigger_args_dict':args.trigger_args_dict,
        'ilp':args.ilp,
    }

    print('构造estimator')
    estimator = tf.estimator.Estimator(model_fn, config=run_config, params=params)

    if(args.isTraining):
        print('获取训练数据。。。')
        train_inpf = functools.partial(input_fn, input_dir=(os.path.join(args.labeled_data_path, 'train')),
                                       shuffe=True, num_epochs=args.num_epochs, batch_size=args.batch_size,
                                       max_sequence_length=args.max_sequence_length,embedded_dim=args.embedded_dim,wv=args.wv,tag2id=args.tag2id)
        train_total = len(list(train_inpf()))
        print('训练steps:' + str(train_total))
        print('获取评估数据。。。')
        eval_inpf = functools.partial(input_fn, input_dir=(os.path.join(args.labeled_data_path, 'dev')),
                                      shuffe=False, num_epochs=args.num_epochs, batch_size=args.batch_size,
                                      max_sequence_length=args.max_sequence_length,embedded_dim=args.embedded_dim,wv=args.wv,tag2id=args.tag2id)
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        dev_total = len(list(eval_inpf()))
        print('评估总数：' + str(dev_total))
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf)
        print('开始训练+评估。。。')
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.isTesting:
        print('获取预测数据。。。')
        test_inpf = functools.partial(input_fn, input_dir=(os.path.join(args.labeled_data_path, 'test')),
                                      shuffe=False, num_epochs=1, batch_size=args.batch_size,
                                      max_sequence_length=args.max_sequence_length,embedded_dim=args.embedded_dim,wv=args.wv,tag2id=args.tag2id)

        #crf转移矩阵
        trans = estimator.get_variable_value('transitions')

        predictions = estimator.predict(input_fn=test_inpf)


        # 取真实的tags
        pred_true = generator_fn(input_dir=(os.path.join(args.labeled_data_path, 'test')),max_sequence_length = args.max_sequence_length,wv=args.wv,tag2id=args.tag2id,noEmbedding=True)
        words_list = [x[0][0] for x in pred_true]
        tags_list = [x[1] for x in pred_true]
        lengths = [x[0][1] for x in pred_true]

        if(args.ilp):
            # blstm获得的结果
            logits = [x['logits'] for x in list(predictions)]

            # 使用ILP-solver优化后的结果
            with open(os.path.join(output_dir,'predict.txt'),'w',encoding='utf-8') as fw:
                for logit,words,tags,length in zip(logits,words_list,tags_list,lengths):
                    scores,ids_list = optimize(length,args.num_labels,trans,logit,args.id2tag,args.trigger_ids,args.trigger_args_dict,args.i_ids)
                    fw.write('原句：'+' '.join(words)+'\n')
                    fw.write('目标标记：'+' '.join(tags)+'\n')
                    index = 0
                    for socre,ids in zip(scores,ids_list):
                        fw.write('预测结果%d:,得分%f,标记：%s \n' % (index,socre,str(' '.join([args.id2tag[id] for id in ids]))))
                        index+=1
        else:
            pred_ids_list = [x['pred_ids'] for x in list(predictions)]
            indices = [item[1] for item in args.tag2id.items() if (item[0] != '<pad>' and item[0] != 'O')]

            report = flat_classification_report(y_pred=pred_ids_list, y_true=tags_list, labels=indices)

            with open(os.path.join(output_dir,'predict.txt'),'w',encoding='utf-8') as fw:
                fw.write(report)
                for pred_ids,words,tags,length in zip(pred_ids_list,words_list,tags_list,lengths):
                    fw.write('原句：'+' '.join(words[0:length])+'\n')
                    fw.write('目标标记：'+' '.join([args.id2tag[id] for id in tags[0:length]])+'\n')
                    fw.write('预测结果：'+(' '.join([args.id2tag[id] for id in pred_ids[0:length]]))+'\n')

if __name__ == '__main__':
    main(getArgs())
    pass
