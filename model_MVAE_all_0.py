import torch
from torch.nn import functional
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter

def trace(A=None, B=None):
    if A is None:
        print('please input pytorch tensor')
        val = None
    elif B is None:
        val = torch.sum(A * A)
    else:
        val = torch.sum(A * B)
    return val


##  MVAE   引入att预训练的结果， 同时引入adj的数据，分层的结构进行预测
class cVAE_Model_0(nn.Module):
    """
    MVAE  + att  +  adj
    """

    def __init__(self, p_dims, q_dims, q_dims_att, q_dims_adj, dropout=0.5):
        super(cVAE_Model_0, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]


        self.q_dims_att = q_dims_att
        self.q_dims_adj = q_dims_adj

        # Last dimension of q- network is for mean and variance
        # q_layers  dim_nums, **, latent,
        # Encode parameters  编码器参数设置
        # 编码 Att 参数
        temp_q_dims_att = self.q_dims_att[:-1] + [self.q_dims_att[-1] * 2]
        self.q_att_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(temp_q_dims_att[:-1], temp_q_dims_att[1:])])

        # 编码 Adj 参数
        self.q_adj_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                           d_in, d_out in zip(self.q_dims_adj[:-1], self.q_dims_adj[1:])])
        self.dense_att_concat = nn.Linear(q_dims[-1] * 2, q_dims[-1] * 2)
        self.dense_attz = nn.Linear(q_dims_att[-1], q_dims_att[-1])

        # 编码 Att + Adj 联同 X 参数
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(q_dims[:-1], q_dims[1:])])
        self.dense_adj_concat = nn.Linear(q_dims[-1] * 2, q_dims[-1] * 2)
        # self.dense_adjz = nn.Linear(q_dims_adj[-1], q_dims_adj[-1])


        # Decoder parameters   解码器参数设置
        # self.dense_decode_z = nn.Linear(q_dims[-1], q_dims[-1])
        # self.dense_decode_zatt = nn.Linear(q_dims[-1], q_dims[-1])
        # self.dense_decode_zadj = nn.Linear(q_dims[-1], q_dims[-1])

        temp_p_dims = [self.p_dims[0]*3] + self.p_dims[1:]
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(temp_p_dims[:-1], temp_p_dims[1:])])


        self.pz_att_layers = nn.Linear(self.p_dims[0], self.p_dims[0] * 2)
        self.pz_adj_layers = nn.Linear(self.p_dims[0], self.p_dims[0] * 2)
        # self.p_dims_att = [self.p_dims[0], self.p_dims[0] * 2]
        # self.pz_att_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims_att[:-1], self.p_dims_att[1:])])

        # self.p_dims_adj = [self.p_dims[0], self.p_dims[0] * 2]
        # self.pz_adj_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
        #                                     d_in, d_out in zip(self.p_dims_adj[:-1], self.p_dims_adj[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input_x_huai, input_att, input_adj, input_x):
        # 属性编码  得到zA     qzA  q(zAt | A, AX)    需要  q(zAt) || p(z)
        att_mu, att_logvar = self.encode_att(input_att)
        att_z = self.reparameterize(att_mu, att_logvar)

        # 邻居编码  得到zA     qzA  q(zAj | zAt, ADJ)  需要 q(zAj) || p'(zAj|zAt)
        adj_mu, adj_logvar = self.encode_adj(input_adj, att_z)
        adj_z = self.reparameterize(adj_mu, adj_logvar)

        # 结合 adj 和序列x得到 latent zX   qzX   q(zX | zAj, X) 需要 q(zX) || p'(zX|zAj)
        x_mu, x_logvar = self.encode(input_x_huai, adj_z)
        z = self.reparameterize(x_mu, x_logvar)

        # zAt zAj 和 zX 共同解码   # 加入非线性层。
        # z = torch.tanh(self.dense_decode_z(z))
        # z_att = torch.tanh(self.dense_decode_zatt(att_z))
        # z_adj = torch.tanh(self.dense_decode_zadj(adj_z))
        # final_dec = torch.cat((z, z_att, z_adj), dim=-1)

        final_dec = torch.cat((z, att_z, adj_z), dim=-1)
        dec_out = self.decode(final_dec)


        # 损失还需要额外的， 即qzX  同 pzX 的KL散度。
        # p'(zAj|zAt)   Adj 同 Att的 KL散度
        xp_att_mu, xp_att_logvar = self.encode_pz_att(att_z)

        # p'(zX|zAj)   X 同 Adj 的KL散度
        xp_adj_mu, xp_adj_logvar = self.encode_pz_adj(adj_z)



        # 增加额外的损失函数，即原始的完整的历史 和 当前生成的z的兴趣分布是一致的。
        # real_mu, real_logvar = self.encode_adj(input_x_all) # 等价于 同adj共享参数
        # 如果想要decode参数的转置呢？
        real_mu, real_logvar = 0, 0
        # print("shape", x_mu.shape, xp_adj_mu.shape, xp_adj_logvar.shape, adj_mu.shape, xp_att_mu.shape)
        return dec_out, x_mu, x_logvar, xp_adj_mu, xp_adj_logvar, adj_mu, adj_logvar, \
               xp_att_mu, xp_att_logvar, real_mu, real_logvar


    def encode_att(self, input):
        h = input
        for i, layer in enumerate(self.q_att_layers):
            h = layer(h)
            if i != len(self.q_att_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims_att[-1]]
                logvar = h[:, self.q_dims_att[-1]:]
        return mu, logvar


    def encode_adj(self, input, att_z):
        h = input
        for i, layer in enumerate(self.q_adj_layers):
            h = layer(h)
            if i != len(self.q_adj_layers) - 1:
                h = F.tanh(h)
            else:
                h = h

        h = torch.tanh(h)
        att_z = torch.tanh(self.dense_attz(att_z))

        concat = torch.cat((h, att_z), dim=-1)  # 2*latent  联合
        concat_final = self.dense_att_concat(concat)

        mu = concat_final[:, :self.q_dims_adj[-1]]
        logvar = concat_final[:, self.q_dims_adj[-1]:]
        return mu, logvar


    def encode(self, input, adj_z):
        # h = F.normalize(input)
        # h = self.drop(h)
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                h = h

        # h = torch.tanh(h)
        # adj_z = torch.tanh(self.dense_adjz(adj_z))

        concat = torch.cat((h, adj_z), dim=-1)  # 2*latent  联合
        concat_final = self.dense_adj_concat(concat)

        mu = concat_final[:, :self.q_dims[-1]]
        logvar = concat_final[:, self.q_dims[-1]:]
        return mu, logvar


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, input):
        h = input
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def encode_pz_att(self, input):
        # h = input
        # for i, layer in enumerate(self.pz_att_layers):
        #     h = layer(h)
        #     if i != len(self.pz_att_layers) - 1:
        #         h = F.tanh(h)
        #     else:
        #         mu = h[:, :self.p_dims[-1]]
        #         logvar = h[:, self.p_dims[-1]:]
        h = self.pz_att_layers(input)
        mu = h[:, :self.p_dims[0]]
        logvar = h[:, self.p_dims[0]:]
        return mu, logvar

    def encode_pz_adj(self, input):
        # h = input
        # for i, layer in enumerate(self.pz_adj_layers):
        #     h = layer(h)
        #     if i != len(self.pz_adj_layers) - 1:
        #         h = F.tanh(h)
        #     else:
        #         mu = h[:, :self.p_dims[-1]]
        #         logvar = h[:, self.p_dims[-1]:]
        h = self.pz_adj_layers(input)
        mu = h[:, :self.p_dims[0]]
        logvar = h[:, self.p_dims[0]:]
        return mu, logvar

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
