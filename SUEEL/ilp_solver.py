#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__doc__ = 'description'
__author__ = '13314409603@163.com'

from gurobipy import *

def optimize(length,num_labels,trans,logits,id2tag,trigger_ids,trigger_args_dict):
    '''

    :param length: 句长
    :param num_labels: 标签数量
    :param trans: crf转移矩阵
    :param logits: blstm经过全连接层生成的置信分数
    :param id2tag: id转标签
    :param trigger_ids: B_触发词类标签id
    :param trigger_args_dict:  每类B_触发词标签对应的B_参数标签id
    :return:
    '''
    try:
        m = Model('ilp-solver')

        x = m.addVars(length-1,num_labels,num_labels,vtype=GRB.BINARY,name="x")

        m.setObjective(sum(x[i,l,h]*(logits[i][l]+trans[l][h])for h in range(num_labels) for l in range(num_labels) for i in range(length-1)),GRB.MAXIMIZE)

        m.addConstrs((x.sum(i,'*','*')==1 for i in range(length-1)),name='con1')
        m.addConstrs( x.sum(i+1,h,'*')==1 for h in range(num_labels) for l in range(num_labels) for i in range(length-2) if x[i,l,h]==1 )
        m.addConstrs((x[i-1,l-1,l]==1 or x[i-1,l,l]==1) for h in range(num_labels) for l in range(num_labels) for i in range(1,length-1) if(x[i,l,h]==1 and id2tag[l].find('I_')!=-1) )

        #更新约束为：必须出现触发词才能出现事件的参数
        for trigger_id in trigger_ids:
            if(trigger_id in trigger_args_dict):
                for arg_id in trigger_args_dict[trigger_id]:
                    m.addConstr(x.sum('*',trigger_id,'*')>=x.sum('*',arg_id,'*'))

        m.optimize()
        ids =  get_result(length,num_labels,x)
        print(ids)
        return ids

    except GurobiError as e:
        print('Error code'+str(e.errno)+':'+str(e))
    except AttributeError as e:
        print('some error:'+str(e))

def get_result(length,num_labelds,x):
    result = []
    for i in range(length-1):
        for l in range(num_labelds):
            for h in range(num_labelds):
                if(x[i,l,h]==1):
                    if(i==0):
                        result.append(l)
                    result.append(h)
    return result

if __name__ == '__main__':
    pass