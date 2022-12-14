
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(file_name):
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + file_name
    df = pd.read_csv(path, encoding='gbk')
    df.fillna(df.mean(), inplace=True)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multiple outputs data processing.
def nn_seq_mo(seq_len, B, num):
    data = load_data('1min__data_newyork_sum_1240.csv')

    train = data[:int(len(data) * 0.8)]
    val = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, step_size, shuffle):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        seq = []
        for i in tqdm(range(0, len(dataset) - seq_len - num, step_size)):
            train_seq = []
            train_label = []
            avg = 0
            for j in range(i, i + num):
                avg += load[j]
                if (j-i+1) % 3==0:
                    train_seq.append([avg/3])
                    avg = 0

            for j in range(i, i + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=num, shuffle=False)

    return Dtr, Val, Dte, m, n


# for j in range(i, i + seq_len):
#     x = [load[j]]
#     # for c in range(1, 8):
#     #     x.append(dataset[j][c])
#     train_seq.append(x)
#
# for j in range(i + seq_len, i + seq_len + num):
#     train_label.append(load[j])


# Single step scrolling data processing.
def nn_seq_sss(seq_len, B):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, shuffle):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        seq = []
        for i in range(len(dataset) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, True)
    Val = process(val, B, True)
    Dte = process(test, B, False)

    return Dtr, Val, Dte, m, n


# Multiple models single step data processing.
def nn_seq_mmss(seq_len, B, pred_step_size):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, step_size, shuffle):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        load = load.tolist()
        #
        seqs = [[] for i in range(pred_step_size)]
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            train_seq = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            for j, ind in zip(range(i + seq_len, i + seq_len + pred_step_size), range(pred_step_size)):
                #
                train_label = [load[j]]
                seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seqs[ind].append((seq, train_label))
        #
        res = []
        for seq in seqs:
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
            res.append(seq)

        return res

    Dtrs = process(train, B, step_size=1, shuffle=True)
    Vals = process(val, B, step_size=1, shuffle=True)
    Dtes = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtrs, Vals, Dtes, m, n


# Multi task learning
def nn_seq_mtl(seq_len, B, pred_step_size):
    data = load_data('mtl_data_1.csv')
    # split
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # normalization
    train.drop([train.columns[0]], axis=1, inplace=True)
    val.drop([val.columns[0]], axis=1, inplace=True)
    test.drop([test.columns[0]], axis=1, inplace=True)
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)

    def process(dataset, batch_size, step_size, shuffle):
        dataset = dataset.tolist()
        seq = []
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  # ???24????????????????????????
                    x.append(dataset[j][c])
                train_seq.append(x)
            # ??????????????????????????????
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            train_labels = torch.FloatTensor(train_labels)
            seq.append((train_seq, train_labels))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtr, Val, Dte, scaler


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
