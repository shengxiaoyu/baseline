#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import tensorflow as tf
import functools
import os

def input_fn(input_dir,shuffe,num_epochs,batch_size,max_sequence_length,embedded_dim,wv,tag2id):
    shapes = (([max_sequence_length,embedded_dim],()),[max_sequence_length])
    types = ((tf.float32,tf.int32),tf.int32)

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, input_dir=input_dir,
                          max_sequence_length=max_sequence_length,wv=wv,tag2id=tag2id),
        output_shapes=shapes,
        output_types=types
    )
    if shuffe:
        dataset = dataset.shuffle(buffer_size=20000).repeat(num_epochs)

    dataset = dataset.batch(batch_size)
    return dataset

def read_file(input_dir):
    result = []
    for input_file in os.listdir(input_dir):
        with open(os.path.join(input_dir, input_file), 'r', encoding='utf8') as f:
            sentence = f.readline()  # 句子行
            while sentence:
                sentence = sentence.strip()
                # 标记行
                label = f.readline()
                #pos行
                f.readline()
                if not label:
                    break
                label = label.strip()
                length = len(sentence.split())

                one_example = [sentence,label,length]
                result.append(one_example)
    return result

def generator_fn(input_dir,max_sequence_length,wv,tag2id):
    result = []
    for input_file in os.listdir(input_dir):
        with open(os.path.join(input_dir, input_file), 'r', encoding='utf8') as f:
            sentence = f.readline()  # 句子行
            while sentence:
                # 标记行
                label = f.readline()
                #pos行
                f.readline()
                if not label:
                    break
                words = sentence.strip().split(' ')
                words = list(filter(lambda word: word != '', words))

                tags = label.strip().split(' ')
                tags = list(filter(lambda word: word != '', tags))

                sentence = f.readline()

                if (len(words) != len(tags)):
                    print(input_file, ' words、labels、pos数不匹配：' + sentence + ' words length:' + str(
                        len(words)) + ' labels length:' + str(len(tags)) )
                    continue
                result.append(paddingAndEmbedding( words, tags, max_sequence_length,wv,tag2id))
    return result

def paddingAndEmbedding(words,tags,max_sequence_length,wv,tag2id):
    # print(fileName)
    length = len(words)

    # 处理意外词
    # 如果是词汇表中没有的词，则使用<pad>代替
    for index in range(length):
        try:
            wv[words[index]]
        except:
            words[index] = '<pad>'
    # padding or cutting
    if (length < max_sequence_length):
        for i in range(length, max_sequence_length):
            words.append('<pad>')
    else:
        words = words[:max_sequence_length]

    if (len(tags) < max_sequence_length):
        for i in range(len(tags), max_sequence_length):
            tags.append('<pad>')
    else:
        tags = tags[:max_sequence_length]


    words = [wv[word] for word in words]

    ids = [tag2id[tag]  for tag in tags]

    return (words, min(length, max_sequence_length)),ids

if __name__ == '__main__':
    pass