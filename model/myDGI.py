import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GAT import GAT
from model.one_side_GCN import oneSide_weighted_rgcn

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class Extract_Overall(nn.Module):
    def __init__(self, opt):
        super(Extract_Overall, self).__init__()
        self.opt = opt
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.weight_matrix = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
    def forward(self, feature, adj):
        h = self.weight_matrix(feature)
        subOutput = torch.mm(adj.to_dense(), h)
        output =self.relu(subOutput)
        subOutput = self.elu(subOutput)
        finalOutput = torch.mean(output, 0)
        return subOutput, finalOutput


class myDGI(nn.Module):
    def __init__(self, opt):
        super(myDGI, self).__init__()
        self.opt = opt
        self.read = AvgReadout()
        self.extract = Extract_Overall(opt)
        self.att = GAT(opt)
        self.sigm = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(opt["hidden_dim"], affine=True)
        self.transform_u = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        self.transform_i = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        self.lin = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.lin_sub = nn.Linear(opt["hidden_dim"] * 2, opt["hidden_dim"])
        # for m in self.modules():
        #     self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    #
    def forward(self, user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out, UV_adj, VU_adj, CUV_adj, CVU_adj, user_One, item_One,
                UV_rated, VU_rated, relation_UV_adj, relation_VU_adj, 
                msk=None, samp_bias1=None,samp_bias2=None):
                
        S_u_One = self.read(user_hidden_out, msk)  # hidden_dim
        S_i_One = self.read(item_hidden_out, msk)  # hidden_dim
        subOutput, Global_item_cor2_user = self.extract(item_hidden_out, UV_rated)
        _, Global_user_cor2_item = self.extract(user_hidden_out, VU_rated) 
        
        g = self.transform_u(torch.cat((S_u_One, Global_item_cor2_user)).unsqueeze(0))
        h = self.transform_i(torch.cat((Global_user_cor2_item, S_i_One)).unsqueeze(0))
        S_Two = g + h   
        S_Two = torch.div(S_Two, 2)
        S_Two = self.sigm(S_Two)  # hidden_dim  need modify
        S_Two = self.lin(S_Two) # 1 * hidden_dim
        S_Two = self.sigm(S_Two)
        
        real_user, real_item = self.att(user_hidden_out, item_hidden_out, UV_adj, VU_adj)
        fake_user, fake_item = self.att(fake_user_hidden_out, fake_item_hidden_out, CUV_adj, CVU_adj)

        real_user_index_feature_Two = torch.index_select(real_user, 0, user_One)
        real_item_index_feature_Two = torch.index_select(real_item, 0, item_One)
        fake_user_index_feature_Two = torch.index_select(fake_user, 0, user_One)
        fake_item_index_feature_Two = torch.index_select(fake_item, 0, item_One)
        real_sub_Two = self.lin_sub(torch.cat((real_user_index_feature_Two, real_item_index_feature_Two),dim = 1))
        real_sub_Two = self.batchnorm1(real_sub_Two)
        real_sub_Two = self.sigm(real_sub_Two)

        fake_sub_Two = self.lin_sub(torch.cat((fake_user_index_feature_Two, fake_item_index_feature_Two),dim = 1))
        fake_sub_Two = self.batchnorm1(fake_sub_Two)
        fake_sub_Two = self.sigm(fake_sub_Two)

        mixup_real = torch.mul(S_Two, real_sub_Two)
        mixup_fake = torch.mul(S_Two, fake_sub_Two)

        return mixup_real, mixup_fake, subOutput, S_Two
