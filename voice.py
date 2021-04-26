import numpy as np
import pandas as pd
import os
import torch
import sys
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

import warnings
warnings.filterwarnings('ignore')
#dic = {label:idx for idx,label in enumerate(np.unique(self.ALL_data['label']))}


class Load_XY(Dataset):
    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class Gender(nn.Module):
    def __init__(self):
        super(Gender, self).__init__()

        self.all = nn.Sequential(
            nn.Linear(20, 100),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.all(x)


class Model():
    def __init__(self, Train_path, Test_path, learning_rate, again=True, flag='未命名'):

        self.learning_rate = learning_rate
        self.again = again
        self.flag = flag
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        self.Gender = Gender().to(self.device)
        self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.Gender.parameters()),
                                lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optim, mode='min', factor=0.4, patience=15, verbose=True, min_lr=0)
        self.loss_fun = nn.BCELoss()

        dic = {'female': 0, 'male': 1}
        if Train_path and not Test_path and again:
            print('用Train_path的数据70%用来训练，30%用来测试，保存为后缀为'+flag+'的pkl文件。')
            self.ALL_data = pd.read_csv(Train_path)
            self.ALL_data['label'] = self.ALL_data['label'].map(dic)
            if 'sound.files' in self.ALL_data.columns:
                X, y = self.ALL_data.iloc[:, 1:-
                                          1].values, self.ALL_data.iloc[:, -1].values
            else:
                X, y = self.ALL_data.iloc[:, :-
                                          1].values, self.ALL_data.iloc[:, -1].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=0)
        elif not Train_path and Test_path and not again:
            print('用后缀为flag的模型对Test_path数据进行测试，生成后缀为self的结果csv文件。')
            self.ALL_data = pd.read_csv(Test_path)
            self.ALL_data['label'] = self.ALL_data['label'].map(dic)
            if 'sound.files' in self.ALL_data.columns:
                self.X_test, self.y_test = self.ALL_data.iloc[:, 1:-
                                                    1].values, self.ALL_data.iloc[:, -1].values
            else:
                self.X_test, self.y_test = self.ALL_data.iloc[:, :-
                                                    1].values, self.ALL_data.iloc[:, -1].values
        elif Train_path and Test_path and again:
            print('用全部Train_path数据训练，用Test_path数据进行测试')
            self.ALL_data = pd.read_csv(Train_path)
            self.ALL_data['label'] = self.ALL_data['label'].map(dic)
            if 'sound.files' in self.ALL_data.columns:
                self.X_train, self.y_train = self.ALL_data.iloc[:, 1:-
                                                      1].values, self.ALL_data.iloc[:, -1].values
            else:
                self.X_train, self.y_train = self.ALL_data.iloc[:, :-
                                                      1].values, self.ALL_data.iloc[:, -1].values
            self.ALL_data = pd.read_csv(Test_path)
            self.ALL_data['label'] = self.ALL_data['label'].map(dic)
            if 'sound.files' in self.ALL_data.columns:
                self.X_test, self.y_test = self.ALL_data.iloc[:, 1:-
                                                    1].values, self.ALL_data.iloc[:, -1].values
            else:
                self.X_test, self.y_test = self.ALL_data.iloc[:, :-
                                                    1].values, self.ALL_data.iloc[:, -1].values

        self.Test_data = DataLoader(Load_XY(self.X_test, self.y_test),
                                    batch_size=256, shuffle=False)

        self.Train_c, self.Val_c, self.Test_c = [], [], []
        self.val_c_best = 0
        self.best_model_test_c = 0

    def Run(self, epoches):
        if self.again:
            self.scheduler.best = float('inf')
            for epoch in range(epoches):
                print(epoch)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X_train, self.y_train, test_size=0.3, random_state=0)
                self.Train_data = DataLoader(
                    Load_XY(X_train, y_train), batch_size=256, shuffle=True)
                self.Val_data = DataLoader(
                    Load_XY(X_val, y_val), batch_size=256, shuffle=True)
                self.Train()
                val_accuracy = self.Val()
                test_accuracy = self.Test()
                if val_accuracy > self.val_c_best:
                    self.val_c_best = val_accuracy
                    self.best_model_test_c = test_accuracy
                    torch.save(self.Gender, 'Gender_lec_'+self.flag+'.pkl')
            result = pd.DataFrame(
                {'train_c': self.Train_c, 'Val_c': self.Val_c, 'Test_c': self.Test_c})
            result.to_csv('result.csv')
            print('val_c_best',self.val_c_best,'best_model_test_c',self.best_model_test_c)
        else:
            self.Gender = torch.load('Gender_lec_'+self.flag+'.pkl')
            self.Test()

    def Train(self):
        self.Gender.train()
        t_pred = []
        t_ori = []
        loss_epoch = 0
        for train_x, train_y in self.Train_data:
            train_x = Variable(train_x.type(
                torch.FloatTensor)).to(self.device)
            train_y = Variable(train_y.type(
                torch.FloatTensor)).to(self.device)
            self.optim.zero_grad()
            y_pre = self.Gender(train_x).squeeze(1)
            loss = self.loss_fun(y_pre, train_y)
            loss.backward()
            self.optim.step()
            y_pre = (y_pre > 0.5)+0
            t_pred.extend(y_pre.data.numpy())
            t_ori.extend(train_y.data.numpy())
            loss_epoch += loss.data.item()
        loss_epoch /= len(t_ori)
        train_accuracy, precision, recall, f1 = metrics(t_pred, t_ori)
        self.Train_c.append(train_accuracy)
        print('\033[1;34m Train:  loss_epoch:\033[0m', loss_epoch)
        print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
            train_accuracy, precision, recall, f1))

    def Val(self):
        self.Gender.eval()
        with torch.no_grad():
            t_pred = []
            t_ori = []
            loss_val = 0
            for test_x, test_y in self.Val_data:
                test_x = Variable(test_x.type(
                    torch.FloatTensor)).to(self.device)
                test_y = Variable(test_y.type(
                    torch.FloatTensor)).to(self.device)

                y_pre = self.Gender(test_x).squeeze(1)
                loss = self.loss_fun(y_pre, test_y)
                y_pre = (y_pre > 0.5)+0
                t_pred.extend(y_pre.data.numpy())
                t_ori.extend(test_y.data.numpy())
                loss_val += loss
            loss_val /= len(t_ori)
            val_accuracy, precision, recall, f1 = metrics(t_pred, t_ori)
            self.scheduler.step(loss_val)
            sys.stdout.flush()
            self.Val_c.append(val_accuracy)
            print('\033[1;34m Val: \033[0m')
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
                val_accuracy, precision, recall, f1))
            return val_accuracy

    def Test(self):
        self.Gender.eval()
        with torch.no_grad():
            t_pred = []
            t_ori = []
            for test_x, test_y in self.Test_data:
                test_x = Variable(test_x.type(
                    torch.FloatTensor)).to(self.device)
                test_y = Variable(test_y.type(
                    torch.FloatTensor)).to(self.device)

                y_pre = self.Gender(test_x).squeeze(1)

                y_pre = (y_pre > 0.5)+0
                t_pred.extend(y_pre.data.numpy())
                t_ori.extend(test_y.data.numpy())
            
            test_accuracy, precision, recall, f1 = metrics(t_pred, t_ori)
            if not self.again:
                if 'sound.files' in self.ALL_data.columns:
                    pre = pd.DataFrame({'file': self.ALL_data.iloc[:, 0].values, 'ori': [
                                       int(x) for x in t_ori], 'pre': t_pred})
                    pre.to_csv('pre_result_'+self.flag+'.csv', index=None)
                else:
                    print('Data中没有sound.files字段，无法生成预测结果csv文件。')
            self.Test_c.append(test_accuracy)
            print('\033[1;34m Test: \033[0m')
            print('\033[1;31m Accuracy:%.4f Precision:%.4f Recall:%.4f F1:%.4f \033[0m' % (
                test_accuracy, precision, recall, f1))
            return test_accuracy


def metrics(results, ori_y):
    '''
        predict: 预测y(需要转化成证整数种类 比如 0  1  2  3 ；若使用二分类 sigmod激活函数，则需要转化一下 比如(out > 0.5)+0)
        ori_y: 原始的01标签
    '''
    accuracy = accuracy_score(ori_y, results)
    precision = precision_score(ori_y, results, labels=[1], average=None)[0]
    recall = recall_score(ori_y, results, labels=[1], average=None)[0]
    f1 = f1_score(ori_y, results, labels=[1], average=None)[0]
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    again = 1
    Train_path = None #'./baidu_bz.csv'  #'./voice.csv'    # './lecvoice_withname.csv'#'./lec_afterfilter.csv'
    Test_path ='./baidu_bz.csv' #None #'./baidu_bz.csv'  # './lec_test.csv'#'./lecvoice_withname.csv'
    flag = 'baidu'
    my_model = Model(Train_path, Test_path, 0.001, again, flag)
    my_model.Run(1500)
