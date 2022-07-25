import os
import sys

sys.path.append('..')
import numpy as np
from torch.nn import functional as F
import pickle

import argparse
import torch
import time
import random
from torch import optim

import matplotlib.pyplot as plt
import numpy as np

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def Wasserstein(mu_1, sigma_1, mu_2, sigma_2):
    p1 = torch.sum(torch.pow((mu_1 - mu_2),2), dim=1)
    # print("sigma", sigma_1.shape, sigma_1)
    # print("sigma", sigma_2.shape, sigma_2)
    p2 = torch.sum(torch.pow(torch.pow(torch.abs(sigma_1),1/2) - torch.pow(torch.abs(sigma_2), 1/2),2) , 1)
    return torch.mean(p1+p2)


def compute_kl(mu1, logvar1, mu2, logvar2):
    """
    计算两个多元高斯分布之间KL散度KL(N1||N2)；  所有的shape均为(B1,B2,...,dim),表示协方差为0的多元高斯分布
    这里我们假设加上Batch_size，即形状为(B,dim)    dim:特征的维度
    """
    numerator = logvar1.exp() + torch.pow(mu1-mu2, 2)
    fraction = torch.div(numerator, logvar2.exp())
    kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction -1, dim=1)
    return kl.mean(dim=0)


def loss_function_2(recon_x, x, x_mu, x_logvar, att_mu, att_logvar, adj_mu, adj_logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # 此时应该是先验分布为att_mu
    KLD = compute_kl(x_mu, x_logvar, att_mu, att_logvar)
    KLD_adj = -0.5 * torch.mean(torch.sum(1 + adj_logvar - adj_mu.pow(2) - adj_logvar.exp(), dim=1))
    return BCE + anneal * (KLD+KLD_adj)

def loss_function_3(recon_x, x, x_mu, x_logvar, att_mu, att_logvar, adj_mu, adj_logvar,
                    real_mu, real_logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    # KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    # 此时应该是先验分布为att_mu
    KLD = compute_kl(x_mu, x_logvar, att_mu, att_logvar)
    KLD_adj = -0.5 * torch.mean(torch.sum(1 + adj_logvar - adj_mu.pow(2) - adj_logvar.exp(), dim=1))
    KLD_real = Wasserstein(x_mu, x_logvar, real_mu, real_logvar)
    print("KLD_real", KLD_real)
    return BCE + anneal * (KLD + KLD_adj + KLD_real)


def loss_function_4(recon_x, x, x_mu, x_logvar, xpadj_mu, xpadj_logvar, adj_mu, adj_logvar,
                    adjtt_mu, adjtt_logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD_adj = compute_kl(x_mu, x_logvar, xpadj_mu, xpadj_logvar)
    KLD_att = compute_kl(adj_mu, adj_logvar, adjtt_mu, adjtt_logvar)
    return BCE + anneal * (KLD_adj + KLD_att)


def loss_function_5(recon_x, x, x_mu, x_logvar, xpadj_mu, xpadj_logvar, adj_mu, adj_logvar,
                    adjtt_mu, adjtt_logvar, real_mu, real_logvar, anneal=1.0):
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD_adj = compute_kl(x_mu, x_logvar, xpadj_mu, xpadj_logvar)
    KLD_att = compute_kl(adj_mu, adj_logvar, adjtt_mu, adjtt_logvar)
    KLD_real = Wasserstein(x_mu, x_logvar, real_mu, real_logvar)
    return BCE + anneal * (KLD_adj + KLD_att + KLD_real)


# dataset_name = 'movieline'  #  movieline crossbook
# adj_num = 3
# data_train_file = "../../train_test_{}/train_test_VAE_{}_att_adj_{}.pkl".format(dataset_name, dataset_name, adj_num)
#
#
# with open(data_train_file, 'rb') as f2:
#     train_set = pickle.load(f2)
#     test_few_set_0 = pickle.load(f2)
#     test_few_set_1 = pickle.load(f2)
#     test_few_set_3 = pickle.load(f2)
#     test_few_set_5 = pickle.load(f2)
#     test_old_set = pickle.load(f2)
#     user_count, item_count = pickle.load(f2)
# print(user_count, item_count, len(train_set), len(test_few_set_5), len(test_old_set))
#
# num_count = []
# for user in train_set.keys():
#     print("line", user)
#     hist = train_set[user][0]
#     print("hist", len(hist), hist)
#     num_count.append(len(hist))
#
# for user in test_few_set_5.keys():
#     print("line", user)
#     hist = test_few_set_5[user][0]
#     num_count.append(len(hist))
#
# print(num_count)
# arr_mean = np.mean(num_count)
# arr_var = np.var(num_count)
# arr_std = np.std(num_count,ddof=1)
# arr_New = sorted(num_count)
# arr_Min = arr_New[0]
# arr_Max = arr_New[len(arr_New)-1]
# arr_Mid = arr_New[int(len(arr_New)/2)]
# print(np.max(num_count), np.min(num_count), arr_Max, arr_Min, arr_mean, arr_var, arr_std, arr_Mid)
# 2313 6 2313 6 156.84615384615384 36173.484023668636 190.21305439519926  86   Movieline
# # 5890 4 5890 4 44.76602327979962 29073.071525706244 170.52083684395208  9   BookCrossing

#
#
# dataset_name = 'crossbook'
# num_num = 5 # 5,6,7,10     6,7 10中选择
# sim_select = 'bav'  # bav, att, both
# adj_num = 3
# data_train_file = "../../train_test_{}/train_test_VAE_{}_att_adj_{}_{}_{}.pkl".format(dataset_name, dataset_name, adj_num, sim_select, num_num)
# with open(data_train_file, 'rb') as f2:
#     train_set = pickle.load(f2)
#     test_few_set_0 = pickle.load(f2)
#     test_few_set_1 = pickle.load(f2)
#     test_few_set_3 = pickle.load(f2)
#     test_old_set = pickle.load(f2)
#     user_count, item_count = pickle.load(f2)
# print(user_count, item_count, len(train_set), len(test_few_set_3), len(test_old_set))
# num_count = []
# for user in train_set.keys():
#     print("line", user)
#     hist = train_set[user][0]
#     print("hist", len(hist), hist)
#     num_count.append(len(hist))
#
# for user in test_few_set_3.keys():
#     print("line", user)
#     hist = test_few_set_3[user][0]
#     num_count.append(len(hist))
#
# print(num_count)
# arr_mean = np.mean(num_count)
# arr_var = np.var(num_count)
# arr_std = np.std(num_count,ddof=1)
# arr_New = sorted(num_count)
# arr_Min = arr_New[0]
# arr_Max = arr_New[len(arr_New)-1]
# arr_Mid = arr_New[int(len(arr_New)/2)]
# print(np.max(num_count), np.min(num_count), arr_Max, arr_Min, arr_mean, arr_var, arr_std, arr_Mid)
# # 5890 4 5890 4 44.76602327979962 29073.071525706244 170.52083684395208

# # 指定分组个数
# num_bins = 100
# fig, ax = plt.subplots()
# # 绘图并接受返回值
# n, bins_limits, patches = ax.hist(num_count, num_bins, density=1)
# # 添加分布曲线
# ax.plot(bins_limits[:100],n,'--')
# plt.title('直方图数据添加分布曲线')
# plt.show()