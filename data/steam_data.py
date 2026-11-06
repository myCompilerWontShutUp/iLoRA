import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

import pandas as pd
import random

class SteamData(data.Dataset):
    def __init__(self, data_dir=r'data/ref/steam',
                 stage=None,
                 cans_num=10,
                 sep=", ",
                 no_augment=True):
        self.__dict__.update(locals())
        self.aug = (stage=='train') and not no_augment
        self.padding_item_id=3581
        self.check_files()

    # 返回session_data['seq']长度
    def __len__(self):
        return len(self.session_data['seq'])

    # 获取索引i的样本(批次i)
    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp['seq_unpad'],temp['next'])
        cans_name=[self.item_id2name[can] for can in candidates]
        sample = {
            'sample_idx': i,
            'seq': temp['seq'],
            'seq_name': temp['seq_title'],
            'len_seq': temp['len_seq'],
            'seq_str': self.sep.join(temp['seq_title']),
            'cans': candidates,
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': self.cans_num,
            'item_id': temp['next'],
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    
    # 进行负采样, 返回序列ID列表
    def negative_sampling(self,seq_unpad,next_item):
        # canset: 所有游戏id中不在seq_unpad中的游戏id
        canset=[i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i!=next_item]
        # 随机选择cans_num-1个游戏id, 加上next_item
        candidates=random.sample(canset, self.cans_num-1)+[next_item]
        random.shuffle(candidates)
        return candidates  

    # 检查并加载数据文件
    def check_files(self):
        self.item_id2name=self.get_game_id2name()
        if self.stage=='train':
            filename="train_data.df"
        elif self.stage=='val':
            filename="Val_data.df"
        elif self.stage=='test':
            filename="Test_data.df"
        data_path=op.join(self.data_dir, filename)
        # 根据data_path和id2name字典加载数据
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  

    # 获取游戏id到游戏名的映射, 返回字典
    def get_game_id2name(self):
        game_id2name = dict()
        item_path=op.join(self.data_dir, 'id2name.txt')
        with open(item_path, 'r') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                game_id2name[int(ll[0])] = ll[1].strip()
        return game_id2name
    
    # 对数据进行预处理
    def session_data4frame(self, datapath, game_id2name):
        # 根据datapath读取pd数据
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data['len_seq'] >= 3]
        # 从序列中移除填充项
        def remove_padding(xx):
            x = xx[:]
            for i in range(10):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x
        # 去除pad的train_data序列 -> train_data['seq_unpad']
        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        # 序列号 -> 游戏名
        def seq_to_title(x): 
            return [game_id2name[x_i] for x_i in x]
        # 转换train_data ID序列为游戏名序列 -> train_data['seq_title']
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)
        # 单个序列 -> 游戏名
        def next_item_title(x): 
            return game_id2name[x]
        # 转换train_data['next'] ID序列为游戏名序列 -> train_data['next_item_name']
        train_data['next_item_name'] = train_data['next'].apply(next_item_title)
        return train_data