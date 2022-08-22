import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GNN
from model.GNN2 import GNN2
from model.AttDGI import AttDGI
from model.pretrained_ml100k import pretrained, genreEncoder, genreDecoder
from model.myDGI import myDGI

class EmbeddingLayer(nn.Module):
    def __init__(self, opt):
        super(EmbeddingLayer, self).__init__()
        self.user_embedding = nn.Embedding(opt["number_user"], opt["feature_dim"])
        self.item_embedding = nn.Embedding(opt["number_item"], opt["feature_dim"])
        self.user_embed = nn.Linear(opt["feature_dim"], opt["hidden_dim"])
        self.item_embed = nn.Linear(opt["feature_dim"], opt["hidden_dim"])
        self.item_index = torch.arange(0, opt["number_item"], 1)
        self.user_index = torch.arange(0, opt["number_user"], 1)
        # print('user0 embedding:')
        # print(self.user_embedding(torch.tensor(0)))
        if opt["cuda"]:
            self.item_index = self.item_index.cuda()
            self.user_index = self.user_index.cuda()  
        self.GNN = GNN(opt)
    
    def forward(self, ufea, vfea, UV_adj, VU_adj, adj):
        # ufea = self.user_embed(ufea)
        # vfea = self.item_embed(vfea)
        learn_user,learn_item = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_user,learn_item    
        
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt=opt
        self.embedding = EmbeddingLayer(opt)
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        self.extractor = myDGI(opt)
        self.dropout = opt["dropout"]
        self.relu = nn.ELU()
    
    def score_predict(self, sub_global_fea, local_fea):
        # print('-' * 20)
        # print('global fea[0][0]:')
        # print(global_fea[0][0])
        # print('global fea[1][0]:')
        # print(global_fea[1][0]) 
        # print('-' * 20)        
        # print('local fea[0][0]:')
        # print(local_fea[0][0])
        # print('local fea[1][0]:')
        # print(local_fea[1][0])        
        out = self.embedding.GNN.score_function1(local_fea) #[128, 1650, 64]
        out = self.relu(out)
        # print('-' * 20)
        # print('after 1st layer NN:')
        # print('out[0][0]:')
        # print(out[0][0])
        # print('out[1][0]:')
        # print(out[1][0])
        # out = self.embedding.GNN.view_function0(torch.cat((global_fea, sub_global_fea + out), dim=-1))
        # out = self.relu(out)        
        out = self.embedding.GNN.view_function1(sub_global_fea + out) 
        # out = self.embedding.GNN.view_function1(torch.cat((global_fea, out), dim=-1))
        # out = self.embedding.GNN.view_function1(global_fea, out)
        out = self.relu(out)
        # print('-' * 20)
        # print('after 2nd layer NN:')    
        # print('out[0][0]:')
        # print(out[0][0])
        # print('out[1][0]:')
        # print(out[1][0])         
        out = self.embedding.GNN.score_function2(out)
        out = self.relu(out)
        out = self.embedding.GNN.score_function3(out)
        out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, sub_global_fea, local_fea):  
        # print('-' * 20)
        # print('global fea[0]:')
        # print(global_fea[0])
        # print('global fea[1]:')
        # print(global_fea[1]) 
        # print('-' * 20)        
        # print('local fea[0]:')
        # print(local_fea[0])
        # print('local fea[1]:')
        # print(local_fea[1])              
        out = self.embedding.GNN.score_function1(local_fea)           
        out = self.relu(out) #(128, 64)
        # print('-' * 20)
        # print('after 1st layer NN:')
        # print('out[0]:')
        # print(out[0])
        # print('out[1]:')
        # print(out[1]) 
        # out = self.embedding.GNN.view_function0(torch.cat((global_fea, sub_global_fea + out), dim=1))
        # out = self.relu(out)
        # out = self.embedding.GNN.view_function1(global_fea + sub_global_fea + out)
        out = self.embedding.GNN.view_function1(sub_global_fea + out)       
        # out = self.embedding.GNN.view_function1(torch.cat((global_fea, out), dim=1))
        # out = self.embedding.GNN.view_function1(global_fea, out)
        out = self.relu(out)
        # print('-' * 20)
        # print('after 2nd layer NN:')    
        # print('out[0]:')
        # print(out[0])
        # print('out[1]:')
        # print(out[1])              
        out = self.embedding.GNN.score_function2(out)          
        out = self.relu(out)     
        out = self.embedding.GNN.score_function3(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    def forward(self, user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out,
                UV, VU, CUV, CVU, user_One, item_One, UV_rated, VU_rated, relation_UV_adj, relation_VU_adj):
        
        return self.extractor(user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out,
                            UV, VU, CUV, CVU, user_One, item_One,
                            UV_rated, VU_rated, relation_UV_adj, relation_VU_adj)
                

class Discriminator(nn.Module):
    def __init__(self, d_model) -> None:
        super(Discriminator, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1)
        self.Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.lin = nn.Linear(d_model, 1)
        self.sigm = nn.Sigmoid()
    def forward(self, vector): #vector: [128, 64+64]
        vector = torch.unsqueeze(vector, 0)
        output = self.Encoder(vector)
        output = torch.squeeze(output, 0)
        output = self.lin(output)
        score = self.sigm(output)
        return score        