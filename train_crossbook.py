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
from Evaluation import Evaluation
from DataReader import DataReader_all
from util import *

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)





def get_test_result(test_reader):
    mae, rmse, hr, ndcg = [], [], [], []
    y_real = np.zeros(100)
    y_real[99] = 1.0
    # true_idx = 99

    # for uu in test_data_train.keys():
    for x_batch, side_att_batch, side_adj_batch, test_item_list in test_reader.iter_test():
        x_batch = torch.tensor(x_batch, dtype=torch.float)
        side_att_batch = torch.tensor(side_att_batch, dtype=torch.float)
        side_adj_batch = torch.tensor(side_adj_batch, dtype=torch.float)

        recon_batch, _, _, _, _, _, _, _, _, _, _ = cVAE_model(x_batch, side_att_batch, side_adj_batch, x_batch)
        print("loss R_loss", recon_batch)

        rating_batch = recon_batch  # recon_batch.view(-1)
        test_item_ii = torch.LongTensor(test_item_list)

        pred_target_value = torch.gather(rating_batch, dim=1, index=test_item_ii)
        pred_target_value = pred_target_value.data.cpu().numpy()  # # 【0.5,0.2,0.6,0.7,0.1。。。】

        # 都是一批数据，然后分开计算
        for line in pred_target_value:
            _mae, _rmse = Evaluation.prediction(y_real, line)
            _hr, _ndcg = Evaluation.ranking(y_real, line, k=10)

            mae.append(_mae)
            rmse.append(_rmse)
            hr.append(_hr)
            ndcg.append(_ndcg)

    return np.mean(mae), np.mean(rmse), np.mean(hr), np.mean(ndcg)


# 设置随机数种子 # 110，  111，  112
setup_seed(110)

dataset_name = 'crossbook'  # zhihu  taobao   movieline   crossbook   movieline2
# few_shot = 5  # 0, 1, 3, 5   #包含0
model_name = "MVAE_all"
load_model = True
para_dong = True

propa = 0.5  # 表示去除一半的item，去除一半的历史记录，
#  如果1.0的话，则表示去除全部的历史记录。  # 如果为0.0的话，表示不去除历史记录
# adj_num = 5  # 0~8   adj_items_100, adj_items_350, adj_items_530, adj_items_550,
#             # adj_items_1020, adj_items_1030, adj_items_2010, adj_items_3010, adj_items_5010
adj_num = 3 # 0~6   1,3,5,10,20,30,50
num_num = 5
# sim_select = 'bav'  # bav, att, both
way = 1  # 0, 1, 2, 3


# 共有四种选择， False 0, True 1  True 2 和 True 3    首先用于确定那种FCL方式好，一旦确定，便不再改变
##  0 表示不加载 额外的损失， 1表示加载adj模块参数， 2表示加载decode模块参数， 3表示自己训练的参数
if way == 0:
    select_extraloss = False
    select_model = 0
elif way == 1:
    select_extraloss = True
    select_model = 1
elif way == 2:
    select_extraloss = True
    select_model = 2
elif way == 3:
    select_extraloss = True
    select_model = 3
elif way == 4:
    select_extraloss = True
    select_model = 4
else:
    print("input way error")

# data_train_file = "../../train_test_{}/train_test_VAE_{}_att_adj_{}_{}.pkl".format(dataset_name, dataset_name, adj_num, sim_select)

data_train_file = "data/train_test_VAE_{}_att_adj_{}_bav_{}.pkl".format(dataset_name, adj_num, num_num)
data_att_file = "data/user_attribute_{}.pkl".format(dataset_name)


with open(data_train_file, 'rb') as f2:
    train_set = pickle.load(f2)
    test_few_set_0 = pickle.load(f2)
    test_few_set_1 = pickle.load(f2)
    test_few_set_3 = pickle.load(f2)
    test_old_set = pickle.load(f2)
    user_count, item_count = pickle.load(f2)
print(user_count, item_count, len(train_set), len(test_few_set_3), len(test_old_set))

config_all = {
    'num_user': user_count,
    'num_item': item_count + 1,
    'pre_train': False,
    # 'pre_matrix_user': user_matrix,
    # 'pre_matrix_item': item_matrix,

    'anneal_cap': 0.2,
    'total_anneal_steps': 200,
    'number_users_to_keep': 1000000000,

    'topk': 10,
    'num_epoch': 200
}


with open(data_att_file, 'rb') as f:
    user_all_attribute = pickle.load(f)
    age_count, location_count = pickle.load(f)
print(age_count, location_count)

config_single = {'dataset': dataset_name,
                 'dropout': 0.8,
                 'train_batch_size': 512,
                 'test_batch_size': 512,
                 'lr': 0.0001,

                 'num_feature': age_count + location_count,
                 'att_count': 2,
                 'index_all': [0, age_count],
                 'num_d': item_count + 1
                 }



train_reader = DataReader_all(config_all, config_single, train_set, test_old_set, propa)
test_old_reader = DataReader_all(config_all, config_single, train_set, test_old_set, propa)
test_new_reader_0 = DataReader_all(config_all, config_single, train_set, test_few_set_0, propa)
test_new_reader_1 = DataReader_all(config_all, config_single, train_set, test_few_set_1, propa)
test_new_reader_3 = DataReader_all(config_all, config_single, train_set, test_few_set_3, propa)
# test_new_reader_5 = DataReader_all(config_all, config_single, train_set, test_few_set_5, propa)




# training model_att.
# p_dims = [128, 128*3, config_all['num_item']]
# if dataset_name == "movieline":
p_dims = [128, 128 * 3, config_all['num_item']]
q_dims = [config_all['num_item'], 128 * 3, 128]
q_dims_att = [config_single['num_feature'], 128 * 2, 128]
q_dims_adj = q_dims  # [config_all['num_item'], 128 * 2, 128]

    # p_dims = [64, 64 * 3, config_all['num_item']]
    # q_dims = [config_all['num_item'], 64 * 3, 64]
    # q_dims_att = [config_single['num_feature'], 64 * 2, 64]
    # q_dims_adj = q_dims  # [config_all['num_item'], 128 * 2, 128]


if select_extraloss:
    if select_model == 1:
        from model_MVAE_all_1 import cVAE_Model_1
        cVAE_model = cVAE_Model_1(p_dims, q_dims, q_dims_att, q_dims_adj)

    elif select_model == 2:
        from model_MVAE_all_2 import cVAE_Model_2
        cVAE_model = cVAE_Model_2(p_dims, q_dims, q_dims_att, q_dims_adj)

    elif select_model == 3:
        from model_MVAE_all_3 import cVAE_Model_3
        cVAE_model = cVAE_Model_3(p_dims, q_dims, q_dims_att, q_dims_adj)

    elif select_model == 4:
        from model_MVAE_all_4 import cVAE_Model_4
        cVAE_model = cVAE_Model_4(p_dims, q_dims, q_dims_att, q_dims_adj)

    else:
        print("input select_model error")
else:
    from model_MVAE_all_0 import cVAE_Model_0
    cVAE_model = cVAE_Model_0(p_dims, q_dims, q_dims_att, q_dims_adj)


if load_model:
    path = 'pre_model/{}_att_mean'.format(dataset_name)
    p_att_dims = [q_dims_att[2], q_dims_att[1], q_dims_att[0]]
    for l in p_att_dims:
        path += '_' + str(l)
    print('load model_att from path: ' + path)

    dict_trained_att = torch.load(path)
    trained_list = list(dict_trained_att.keys())  # # pre_att_model的参数
    # cVAE_dict = cVAE_model.state_dict()
    # target_list = list(cVAE_dict.keys())  # cVAE_model的参数
    cVAE_model.load_state_dict(dict_trained_att, strict=False)

    # # 冻结 side 层的参数
    if para_dong:
        for name, param in cVAE_model.named_parameters():
            if name in trained_list:
                param.requires_grad = False


# optimizer = optim.Adam(cVAE_model.parameters(), lr=config_single['lr'])
optimizer = optim.Adam(filter(lambda p: p.requires_grad, cVAE_model.parameters()), lr=config_single['lr'])


train_batch_size = config_single['train_batch_size']
num_epoch = config_all['num_epoch']  #

best_HR = 0.0
last_improved = 0
require_improvement = 8
flag = False

global update_count
update_count = 0

for epoch in range(num_epoch):  # 第一次迭代的 all data
    cVAE_model.train()
    loss_value, mae, rmse = [], [], []
    hr, ndcg = [], []
    start = time.time()

    for x_huai_batch, side_att_batch, side_adj_batch, x_batch in train_reader.iter_train():
        # print("x_batch", x_batch)
        if config_all['total_anneal_steps'] > 0:
            anneal = min(config_all['anneal_cap'],
                         1. * update_count / config_all['total_anneal_steps'])
        else:
            anneal = config_all['anneal_cap']
        # print("len(x_batch)", len(x_batch[0]))
        x_huai_batch = torch.tensor(x_huai_batch, dtype=torch.float)
        side_att_batch = torch.tensor(side_att_batch, dtype=torch.float)
        side_adj_batch = torch.tensor(side_adj_batch, dtype=torch.float)
        x_batch = torch.tensor(x_batch, dtype=torch.float)

        # cVAE_Model.zero_grad()
        optimizer.zero_grad()
        recon_batch, x_mu, x_logvar, xpadj_mu, xpadj_logvar, adj_mu, adj_logvar, \
        adjtt_mu, adjtt_logvar, real_mu, real_logvar = cVAE_model(x_huai_batch, side_att_batch,
                                                                  side_adj_batch, x_batch)

        if select_extraloss:
            loss = loss_function_5(recon_batch, x_batch, x_mu, x_logvar, xpadj_mu, xpadj_logvar,
                                   adj_mu, adj_logvar, adjtt_mu, adjtt_logvar, real_mu, real_logvar,
                                   anneal)
        else:
            loss = loss_function_4(recon_batch, x_batch, x_mu, x_logvar, xpadj_mu, xpadj_logvar,
                                   adj_mu, adj_logvar, adjtt_mu, adjtt_logvar, anneal)

        print("loss R_loss", recon_batch, loss)
        loss.backward()
        optimizer.step()

        loss_value.append(loss.item())
        update_count += 1

    print('epoch: {}, loss: {:.6f}, train cost time: {:.1f}s'.
          format(epoch, np.mean(loss_value), time.time() - start))

    # if _ % 1 == 0 and _ != 0:
    print('*****************  testing model_all...  ***************')
    cVAE_model.eval()
    begain = time.time()
    test_MAE_few_0, test_RMSE_few_0, test_HR_few_0, test_NDCG_few_0 = \
        get_test_result(test_new_reader_0)
    test_MAE_few_1, test_RMSE_few_1, test_HR_few_1, test_NDCG_few_1 = \
        get_test_result(test_new_reader_1)
    test_MAE_few_3, test_RMSE_few_3, test_HR_few_3, test_NDCG_few_3 = \
        get_test_result(test_new_reader_3)
    # test_MAE_few_5, test_RMSE_few_5, test_HR_few_5, test_NDCG_few_5 = \
    #     get_test_result(test_new_reader_5)
    test_MAE_old, test_RMSE_old, test_HR_old, test_NDCG_old = \
        get_test_result(test_old_reader)

    print('old mae: {:.5f}, rmse: {:.5f}, hr@10: {:.5f}, ndcg@10: {:.5f}, test cost time: {:.1f}s'.
          format(test_MAE_old, test_RMSE_old, test_HR_old, test_NDCG_old, time.time() - begain))
    print('new_0 mae: {:.5f}, rmse: {:.5f}, hr@10: {:.5f}, ndcg@10: {:.5f}'.
          format(test_MAE_few_0, test_RMSE_few_0, test_HR_few_0, test_NDCG_few_0))
    print('new_1 mae: {:.5f}, rmse: {:.5f}, hr@10: {:.5f}, ndcg@10: {:.5f}'.
          format(test_MAE_few_1, test_RMSE_few_1, test_HR_few_1, test_NDCG_few_1))
    print('new_3 mae: {:.5f}, rmse: {:.5f}, hr@10: {:.5f}, ndcg@10: {:.5f}'.
          format(test_MAE_few_3, test_RMSE_few_3, test_HR_few_3, test_NDCG_few_3))
    # print('new_5 mae: {:.5f}, rmse: {:.5f}, hr@10: {:.5f}, ndcg@10: {:.5f}'.
    #       format(test_MAE_few_5, test_RMSE_few_5, test_HR_few_5, test_NDCG_few_5))

    f11 = open('model_{}_{}_{}_{}_{}_{}_{}_bav_{}.txt'.format(model_name, dataset_name, load_model, para_dong, propa, adj_num, select_model, num_num), 'a+')
    # f11.write(str(epoch) + '\t' + str(train_hr) + '\t' + str(train_ndcg) + '\t' + str(train_mae)
    #           + '\t' + str(train_rmse) + '\n')  #  + '\t' + str(np.mean(loss))
    f11.write(str(epoch) + '\t' + str(test_HR_old) + '\t' + str(test_NDCG_old) + '\t' + str(test_MAE_old)
              + '\t' + str(test_RMSE_old) + '\n')  # + '\t' + str(test_loss_old)
    f11.write(str(epoch) + '\t' + str(test_HR_few_0) + '\t' + str(test_NDCG_few_0) + '\t' + str(test_MAE_few_0)
              + '\t' + str(test_RMSE_few_0) + '\n')  # + '\t' + str(test_loss_few)
    f11.write(str(epoch) + '\t' + str(test_HR_few_1) + '\t' + str(test_NDCG_few_1) + '\t' + str(test_MAE_few_1)
              + '\t' + str(test_RMSE_few_1) + '\n')  # + '\t' + str(test_loss_few)
    f11.write(str(epoch) + '\t' + str(test_HR_few_3) + '\t' + str(test_NDCG_few_3) + '\t' + str(test_MAE_few_3)
              + '\t' + str(test_RMSE_few_3) + '\n')  # + '\t' + str(test_loss_few)
    # f11.write(str(epoch) + '\t' + str(test_HR_few_5) + '\t' + str(test_NDCG_few_5) + '\t' + str(test_MAE_few_5)
    #           + '\t' + str(test_RMSE_few_5) + '\n')  # + '\t' + str(test_loss_few)

    # 如果验证精度提升了，就替换为最好的结果，并保存模型
    if test_HR_old > best_HR:
        best_HR = test_HR_old
        last_improved = epoch
        improved_str = 'improved!'
        print('saving model_att...')
        # path = 'model/{}_MVAE_all_{}_{}_{}_{}_{}_{}'.format(dataset_name, load_model, para_dong, propa, adj_num, select_model, num_num)
        # for l in p_dims:
        #     path += '_' + str(l)
        # cVAE_model.cpu()
        # torch.save(cVAE_model.state_dict(), path)
    else:
        improved_str = ''


    # 如果1000步以后还没提升，就中止训练。
    print("last_improved", epoch, last_improved, require_improvement)
    if epoch - last_improved > require_improvement:
        print("No optimization for ", require_improvement, " steps, auto-stop in the ", epoch, " step!")
        # 跳出这个轮次的循环
        flag = True
        break
    # 跳出所有训练轮次的循环
    if flag:
        break




