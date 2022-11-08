import os
import sys

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process import MyDataset

df = pd.read_csv(r'C:\Users\Caidudu\PycharmProjects\LSTM-MultiStep-Forecasting\data\1min__data_newyork_sum_1240.csv')
# seq_label = pd.read_csv(r'C:\Users\Caidudu\PycharmProjects\LSTM-MultiStep-Forecasting\data\5mins_data_sum_newyork_1240.csv')

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import seq2seq_args_parser, my_seq2seq_args_parser
from model_train import seq2seq_train, load_data
from model_test import seq2seq_test


path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/seq2seq.pkl'

if __name__ == '__main__':
    args = my_seq2seq_args_parser()
    flag = 'seq2seq'

    Dtr, Val, Dte, m, n = load_data(args, flag, args.batch_size)
    print("finish")
    seq2seq_train(args, Dtr, Val, LSTM_PATH)
    seq2seq_test(args, Dte, LSTM_PATH, m, n)