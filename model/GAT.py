import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GAT(nn.Module):
    def __init__(self, opt):
        super(GAT, self).__init__()
        self.att = Attention(opt)
        self.dropout = opt["dropout"]
        self.leakyrelu = nn.LeakyReLU(opt["leakey"])

    def forward(self, ufea, vfea, UV_adj, VU_adj, adj=None):
        learn_user = ufea
        learn_item = vfea

        learn_user = F.dropout(learn_user, self.dropout, training=self.training)
        learn_item = F.dropout(learn_item, self.dropout, training=self.training)
        learn_user, learn_item = self.att(learn_user, learn_item, UV_adj, VU_adj)

        return learn_user, learn_item

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention, self).__init__()
        self.lin_u = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.lin_v = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.divide = int(opt["hidden_dim"] / 2)
        self.u_head1 = nn.Linear(opt["hidden_dim"], self.divide)
        self.u_head2 = nn.Linear(opt["hidden_dim"], self.divide)
        self.v_head1 = nn.Linear(opt["hidden_dim"], self.divide)
        self.v_head2 = nn.Linear(opt["hidden_dim"], self.divide)
        self.relu = nn.ReLU()
        self.opt = opt 

    def forward(self, user, item, UV_adj, VU_adj):
        # user = self.lin_u(user)
        user1 = self.u_head1(user)
        user1 = self.relu(user1)
        user2 = self.u_head2(user)
        user2 = self.relu(user2)
        user = torch.cat((user1, user2), 1)
        # item = self.lin_v(item)
        item1 = self.v_head1(item)
        item1 = self.relu(item1)
        item2 = self.v_head2(item)
        item2 = self.relu(item2)
        item = torch.cat((item1, item2), 1)

        query = user
        key = item
        # import pdb
        # pdb.set_trace()

        value = torch.mm(query, key.transpose(0,1)) # user * item
        value = UV_adj.to_dense()*value  # user * item fuck pytorch!!!
        value /= math.sqrt(self.opt["hidden_dim"]) # user * item
        value = F.softmax(value,dim=1) # user * item
        learn_user = torch.matmul(value,key) + user

        query = item
        key = user
        value = torch.mm(query, key.transpose(0,1))  # item * user
        value = VU_adj.to_dense()*value  # item * user
        value /= math.sqrt(self.opt["hidden_dim"])  # item * user
        value = F.softmax(value, dim=1)  # item * user
        learn_item = torch.matmul(value, key) + item

        return learn_user, learn_item