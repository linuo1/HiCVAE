import tqdm
import numpy as np
import torch
import random

class DataReader_all:
    def __init__(self, config_all, config_single, data_train, data_test, propa):
        self.config_all = config_all
        self.congfig_single = config_single
        self.train_batch_size = config_single['train_batch_size']
        self.test_batch_size = config_single['test_batch_size']
        self.data_train = data_train
        self.data_test = data_test
        self.num_items = config_all['num_item']
        self.num_features = config_single['num_feature']
        self.att_count = config_single['att_count']
        self.index_all = config_single['index_all']
        self.propa = propa



    def iter_train(self):

        users_done = 0

        user_iterate_order = self.data_train.keys()
        # print("user_iterate_order", len(user_iterate_order), user_iterate_order)
        user_iterate_order = list(user_iterate_order)
        # Randomly shuffle the training order
        random.shuffle(user_iterate_order)

        self.epoch_size = len(user_iterate_order) // self.train_batch_size
        # if self.i == self.epoch_size - 1:
        #     raise StopIteration

        for i in range(self.epoch_size):
            ts = user_iterate_order[i * self.train_batch_size: min((i + 1) * self.train_batch_size,
                                                                        len(user_iterate_order))]

            x_batch = []
            x_huai_batch = []
            side_att_batch = []
            side_adj_batch = []
            for user in ts:
                # print("user", user)
                if users_done > self.config_all['number_users_to_keep']: break
                users_done += 1

                # 完整的historical
                hist = self.data_train[user][0]
                xx_bat = np.zeros(self.num_items)
                for item in hist:  # 0 historical ,1 att, 2 adj
                    xx_bat[item] = 1
                x_batch.append(xx_bat)

                # 毁坏的历史
                choice_num = int(self.propa * len(hist))
                drop_items = np.random.choice(hist, size=choice_num)  # 选择为丢弃的数据
                # 其余的 数据，用以恢复重构的数据
                xx_huai = list(set(hist).difference(set(drop_items)))  # b中有而a中没有的
                xx_huai_bat = np.zeros(self.num_items)
                for item in xx_huai:  # 0 historical ,1 att, 2 adj
                    xx_huai_bat[item] = 1
                x_huai_batch.append(xx_huai_bat)


                # 用户属性信息
                att_category = np.zeros(self.num_features)
                for i_c in range(self.att_count):
                    categ = self.data_train[user][1][i_c]
                    ind = self.index_all[i_c]
                    # print("categ, ind", i_c, categ, ind)
                    att_category[categ + ind] = 1
                # print("att_category", att_category)
                side_att_batch.append(att_category)

                # 邻居信息
                xx_bat_adj = np.zeros(self.num_items)  #
                for item in self.data_train[user][3]:  # 0 historical ,1 att, 2 adj
                    xx_bat_adj[item] = 1
                side_adj_batch.append(xx_bat_adj)

            if len(x_batch) == self.train_batch_size:  # batch_size always = 1
                # print("x_batch, side_batch", len(x_batch), len(side_batch), len(x_batch[0]), len(side_batch[0]))
                yield x_huai_batch, side_att_batch, side_adj_batch, x_batch         #, test_item_list



    def iter_test(self):

        users_done = 0

        user_iterate_order = self.data_test.keys()
        # print("user_iterate_order", len(user_iterate_order), user_iterate_order)
        user_iterate_order = list(user_iterate_order)
        # Randomly shuffle the training order
        random.shuffle(user_iterate_order)
        # print("user_iterate_order", user_iterate_order)

        self.epoch_size = len(user_iterate_order) // self.test_batch_size
        # if self.i == self.epoch_size - 1:
        #     raise StopIteration

        for i in range(self.epoch_size):
            ts = user_iterate_order[i * self.test_batch_size: min((i + 1) * self.test_batch_size,
                                                                   len(user_iterate_order))]

            x_batch = []
            side_att_batch = []
            side_adj_batch = []
            test_item_list_all = []
            for user in ts:
                # print("user", user)
                if users_done > self.config_all['number_users_to_keep']: break
                users_done += 1

                # 完整的historical
                hist = self.data_test[user][0][:-1]
                xx_bat = np.zeros(self.num_items)
                for item in hist:  # 0 historical ,1 att, 2 adj
                    xx_bat[item] = 1
                x_batch.append(xx_bat)

                # 属性信息
                att_category = np.zeros(self.num_features)
                for i_c in range(self.att_count):
                    categ = self.data_test[user][1][i_c]
                    ind = self.index_all[i_c]
                    att_category[categ + ind] = 1
                side_att_batch.append(att_category)

                # 邻居信息
                xx_bat_adj = np.zeros(self.num_items)  #
                for item in self.data_test[user][3]:  # 0 historical ,1 att, 2 adj
                    xx_bat_adj[item] = 1
                side_adj_batch.append(xx_bat_adj)


                test_item_list = []
                # test_item_list.append(self.data_test[user][-1])
                # print(set(self.data_test[user]))
                try:
                    # print(set(self.data_train[user]))
                    unobsv_list = list(
                        set(range(self.num_items)) - set(self.data_test[user][0]) - set(self.data_train[user][0]))
                except Exception:
                    unobsv_list = list(set(range(self.num_items)) - set(self.data_test[user][0]))

                neg_samp_list = np.random.choice(unobsv_list, size=99, replace=False)
                for ns in neg_samp_list:
                    test_item_list.append(ns)
                test_item_list.append(self.data_test[user][0][-1])
                test_item_list_all.append(test_item_list)


            if len(x_batch) == self.test_batch_size:  #
                yield x_batch, side_att_batch, side_adj_batch, test_item_list_all  #




