#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import argparse
import os

import numpy as np
from gensim.models import Word2Vec

WV = None
TRIGGER_TAGs = None
ARGU_TAGs =None
TAG_2_ID = None
ID_2_TAG = None
TAGs_LEN = 0
# 分词器
STOP_WORDS=None
TRIGGER_WORDS_DICT = None
TRIGGER_IDS=set()
TRIGGER_ARGS_DICT={}
I_IDS=set()

#初始化各类模型以及词集
def init(rootdir):
    initTags(os.path.join(rootdir,'full_trigger_labels.txt'),os.path.join(rootdir, 'full_argu_labels.txt'))
    initWord2Vec(os.path.join(os.path.join(rootdir, 'newWord2vec'),'word2vec.model'))

def initTags(triggerLablePath,argumentLabelPath):
    global TAG_2_ID, ID_2_TAG,TAGs_LEN,TRIGGER_TAGs,ARGU_TAGs,TRIGGER_IDS,TRIGGER_ARGS_DICT
    TAG_2_ID={}
    ID_2_TAG={}
    TRIGGER_TAGs=[]
    ARGU_TAGs = []
    # 把<pad>也加入tag字典
    TAG_2_ID['<pad>'] = 0
    ID_2_TAG[0] = '<pad>'

    TAG_2_ID['O'] = 1
    ID_2_TAG[1] ='O'
    # 读取根目录下的labelds文件生成tag—id
    index = 2

    # 获取触发词tag
    with open(triggerLablePath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            tag = line.strip()
            TAG_2_ID[tag] = index
            ID_2_TAG[index] = tag
            TRIGGER_TAGs.append(tag)
            if (tag.find('B_') != -1 and tag.find('FamilyConflict')==-1):
                TRIGGER_IDS.add(index)
            if(tag.find('I_')!=-1):
                I_IDS.add(index)
            index += 1

    #获取参数tag
    with open(argumentLabelPath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            tag = line.strip()
            TAG_2_ID[tag] = index
            ID_2_TAG[index] = tag
            ARGU_TAGs.append(tag)
            # 获取触发词
            index1 = tag.find('_')
            index2 = tag.find('_', index1 + 1)
            # 构造触发词id和对应参数id的map
            if (index1 != -1 and index2 != -1):
                trigger = 'B_' + tag[index1 + 1:index2] + '_Trigger'
                trigger_id = TAG_2_ID[trigger]
                if (trigger_id not in TRIGGER_ARGS_DICT):
                    TRIGGER_ARGS_DICT[trigger_id] = set()
                TRIGGER_ARGS_DICT[trigger_id].add(index)
            if (tag.find('I_') != -1):
                I_IDS.add(index)
            index += 1

    TAGs_LEN = len(TAG_2_ID)
def initWord2Vec(word2vec_model_path):
    global WV
    WV = Word2Vec.load(word2vec_model_path).wv
    # <pad> -- <pad> fill word2vec and tags，添加一个<pad>-向量为0的，用于填充
    WV.add('<pad>', np.zeros(WV.vector_size))


def getArgs():
    rootPath = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    # rootPath = '/root/lstm_crf/data'
    init(rootPath)
    ltpPath = os.path.join(rootPath, 'ltp_data_v3.4.0')
    parser = argparse.ArgumentParser(description='Bi-LSTM+CRF')
    parser.add_argument('--root_dir', help='root dir', default=rootPath)
    parser.add_argument('--isTraining', help='train and dev', default=True)
    parser.add_argument('--isTesting', help='test', default=True)
    parser.add_argument('--dropout_rate', help='dropout rate', default=0.5)
    parser.add_argument('--learning_rate', help='learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='hidden units', default=100)
    parser.add_argument('--num_layers', help='num of layers', default=1)
    parser.add_argument('--labeled_data_path', help='labeled data path',
                        default=os.path.join(os.path.join(rootPath, 'labeled'), 'Spe'))
    parser.add_argument('--max_sequence_length', help='max length of sequence', default=55)
    parser.add_argument('--batch_size', help='batch size', default=64)
    parser.add_argument('--num_epochs', help='num of epochs', default=15)
    parser.add_argument('--device_map', help='which device to see', default='CPU:0')
    parser.add_argument('--embedded_dim', help='wordembeddeddim', default=200)
    parser.add_argument('--wv',help='word2vec',default=WV)
    parser.add_argument('--num_labels',help='num of label types',default=TAGs_LEN)
    parser.add_argument('--id2tag',default=ID_2_TAG)
    parser.add_argument('--tag2id',default=TAG_2_ID)
    parser.add_argument('--trigger_ids',default=TRIGGER_IDS)
    parser.add_argument('--trigger_args_dict',default=TRIGGER_ARGS_DICT)
    parser.add_argument('--i_ids',default=I_IDS)
    parser.add_argument('--ilp',default=False)
    args,_ = parser.parse_known_args()
    return args



if __name__ == '__main__':
    pass
