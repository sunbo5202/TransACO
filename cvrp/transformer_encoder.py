import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-6

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain)


class Attention(nn.Module):

    def __init__(self, n_embed, n_head, masked=False):
        super(Attention, self).__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        assert n_embed % n_head == 0
        self.masked = masked

        self.query = init_(nn.Linear(n_embed, n_embed))
        self.key = init_(nn.Linear(n_embed, n_embed))
        self.value = init_(nn.Linear(n_embed, n_embed))
        self.proj = init_(nn.Linear(n_embed, n_embed))

    def forward(self, q, k, v, d=None, mask_ratio=0.1):
        if len(q.shape) == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        B, N, E = q.shape

        q = self.query(q).view(B, N, self.n_head, E // self.n_head).permute(0, 2, 1, 3).contiguous()  # B, H, N, E//H
        k = self.key(k).view(B, N, self.n_head, E // self.n_head).permute(0, 2, 1, 3).contiguous()  # B, H, N, E//H
        v = self.value(v).view(B, N, self.n_head, E // self.n_head).permute(0, 2, 1, 3).contiguous()  # B, H, N, E//H

        # B, H, N, N
        att = (q @ k.transpose(-1, -2).contiguous()) * (1.0 / math.sqrt(k.size(-1)))

        # if self.training:
        #     m_r = torch.ones_like(att) * mask_ratio  # B, H, N, N
        #     att = att + torch.bernoulli(m_r) * 1e-12

        if d is not None:
            # D: B, N, N   dist matrix
            # GraphDistAtt Y = Φ(D)*softmax(QK^T/sqrt(d)+Φ(D))V
            att = att  # B, H, N, N

        if self.masked:
            att = att.masked_fill(self.masked == 0, -1e9)
            att = F.softmax(att, dim=-1)
        else:
            att = F.softmax(att, dim=-1)

        x = att @ v  # B, H, N, E//H
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, E)  # B, N, E

        x = self.proj(x)

        return x


class EncoderBlock(nn.Module):

    def __init__(self, n_embed, n_head, normalization="none", masked=None):
        super(EncoderBlock, self).__init__()
        self.n_embed = n_embed
        self.n_head = n_head
        self.masked = masked
        self.normalization = normalization
        self.att = Attention(n_embed, n_head, masked)

        if normalization == "layer":
            self.norm1 = nn.LayerNorm(n_embed)
            self.norm2 = nn.LayerNorm(n_embed)
        elif normalization == "batch":
            self.norm1 = nn.BatchNorm1d(n_embed)
            self.norm2 = nn.BatchNorm1d(n_embed)
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embed, n_embed * 4)),
            nn.ReLU(),
            init_(nn.Linear(n_embed * 4, n_embed))
        )

    def forward(self, x, dist=None, cosine_sim=None):
        # B N E
        shape = x.shape
        if self.normalization == "layer":
            x = self.norm1(x + self.att(x, x, x) + self.att(x,x,x))
            x = self.norm2(x + self.mlp(x))
        elif self.normalization == "batch":
            x = self.norm1((x + self.att(x, x, x) + self.att(x, x, x)).view(-1, self.n_embed)).view(*shape)
            x = self.norm2((x + self.mlp(x)).view(-1, self.n_embed)).view(*shape)
        else:
            x= x + self.att(x, x, x) ++ self.att(x, x, x)
            x = x + self.mlp(x)
        return x


class Net(nn.Module):

    def __init__(self, n_layers=6, n_embed=128, node_dim=3, n_head=8, normalization="batch", masked=None):
        super(Net, self).__init__()
        self.proj = nn.Linear(node_dim, n_embed)
        self.depot_proj = nn.Linear(2, n_embed)
        self.encoder = nn.Sequential(*(
            EncoderBlock(n_embed, n_head, normalization, masked=masked) for _ in range(n_layers)
        ))
        self.qk_proj = nn.Linear(n_embed, 3 * n_embed)
        # self.layer_norm = nn.LayerNorm([500, 500])  # Add LayerNorm layer
        # init
        # for name, param in self.named_parameters():
        #     stdv = 1. / math.sqrt(param.size(-1))
        #     param.data.uniform_(-stdv, stdv)

    def forward(self, pos, demand, dist=None):
        # pos: (1+N, 2) (index_of_depot:0)
        # demand: (1+N)
        # dist: (1+N, 1+N)
        pos = pos.to(torch.float32)
        demand = demand.to(torch.float32)
        dist = dist.to(torch.float32)
        x = pos
        N = x.size(-2) # depot included, the # of node is (1+n)
        #print(x.shape, demand.shape, dist.shape)
        # import ipdb
        # ipdb.set_trace()
        if dist is None:
            # dist = torch.sqrt(torch.sum((x.unsqueeze(-2) - x.unsqueeze(-3)) ** 2, dim=-1))  # B N N
            # dist[torch.arange(N), torch.arange(N)] = 1e9
            dist = torch.norm(x[:, None] - x, dim=2, p=2)
            dist[torch.arange(N), torch.arange(N)] = 1e9
        node_feature = torch.cat((pos, demand.unsqueeze(-1)), dim=-1)  #(1+N, 3)
        node_embed = self.proj(node_feature[1:, :])  #(N, n_embed)
        depot_embed = self.depot_proj(node_feature[ 0, :2])  #(1, n_embed)
        x = torch.cat((depot_embed.unsqueeze(0), node_embed), dim=-2) #(1+N, n_embed)
        # x: B N E
        for block in self.encoder:
            x = block(x)
        q, k_phe, k_heu = self.qk_proj(x).chunk(3, dim=-1)  # B N E
        # pm = q @ k_phe.transpose(-1, -2) * (1.0 / math.sqrt(k_phe.size(-1))) # phe_mat
        # hm = q @ k_heu.transpose(-1, -2) * (1.0 / math.sqrt(k_heu.size(-1))) # heu_mat
        # pm = torch.tanh(pm)*10
        # hm = torch.tanh(hm)*10
        # # pm = self.layer_norm(pm)
        # # hm = self.layer_norm(hm)
        # # x: B N N
        # #pm = pm - 30 * dist
        # #return 1/ (hm.exp() + EPS)
        # return F.softmax(pm, dim=-1)*2 + EPS, F.softmax(hm, dim=-1)*2 + 1 / dist
        att_phe = q @ k_phe.transpose(-1, -2) * (1.0 / math.sqrt(k_phe.size(-1)))  # phe_mat
        att_heu = q @ k_heu.transpose(-1, -2) * (1.0 / math.sqrt(k_heu.size(-1)))  # heu_mat
        # x: B N N
        # hm = hm + 3 * dist
        # return 1/ (hm.exp() + EPS)
        pm = F.softmax(att_phe, dim=-1) + EPS
        hm = F.softmax(att_heu, dim=-1) + 1 / dist

        # _, indices = torch.topk(dist, 95, -1)
        # hm = torch.scatter(hm, -1, indices, 0)
        # pm = torch.scatter(pm, -1, indices, 0)
        # return pm + EPS, hm + EPS
        return pm, hm + EPS

    @staticmethod
    def reshape(pyg, vector):
        # for compatibility
        return vector
