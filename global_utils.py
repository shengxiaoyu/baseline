#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

import os
from pyltp import Segmentor

rootPath = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
ltpPath = os.path.join(rootPath, 'ltp_data_v3.4.0')
#分词器
SEGMENTOR = Segmentor()
#初始化词性标注模型
SEGMENTOR.load_with_lexicon(os.path.join(ltpPath,'cws.model'), os.path.join(ltpPath,'userDict.txt'))
if __name__ == '__main__':
    pass