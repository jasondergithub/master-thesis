import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.BiGI import BiGI
from model.Model import Generator, Discriminator

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.gen.load_state_dict(checkpoint['gen'])
        self.dis.load_state_dict(checkpoint['dis'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'gen': self.gen.state_dict(),
            'dis': self.dis.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class DGITrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.gen = Generator(opt)
        self.dis = Discriminator(opt["hidden_dim"])
        # self.model = BiGI(opt)
        self.criterion = nn.BCELoss()
        if opt['cuda']:
            self.gen.cuda()
            self.dis.cuda()
            self.criterion.cuda()
        self.optim_G = torch_utils.get_optimizer(opt['optim'], self.gen.parameters(), opt['lr'])
        self.optim_D = torch_utils.get_optimizer(opt['optim'], self.dis.parameters(), opt['lr'])
        self.epoch_rec_loss = []
        self.epoch_dgi_loss = []

    def unpack_batch_predict(self, batch, cuda):
        batch = batch[0]
        if cuda:
            user_index = batch.cuda()
        else:
            user_index = batch
        return user_index

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            user_index = inputs[0]
            item_index = inputs[1]
            negative_item_index = inputs[2]
        return user_index, item_index, negative_item_index

    def unpack_batch_DGI(self, batch, cuda):
        if cuda:
            user_index = batch[0].cuda()
            item_index = batch[1].cuda()
            negative_item_index = batch[2].cuda()
            User_index_One = batch[3].cuda()
            Item_index_One = batch[4].cuda()
            real_user_index_id_Two = batch[5].cuda()
            fake_user_index_id_Two = batch[6].cuda()
            real_item_index_id_Two = batch[7].cuda()
            fake_item_index_id_Two = batch[8].cuda()
        else:
            user_index = batch[0]
            item_index = batch[1]
            negative_item_index = batch[2]
            User_index_One = batch[3]
            Item_index_One = batch[4]
            real_user_index_id_Two = batch[5]
            fake_user_index_id_Two = batch[6]
            real_item_index_id_Two = batch[7]
            fake_item_index_id_Two = batch[8]
        return user_index, item_index, negative_item_index, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two

    def predict(self, batch):
        User_One = self.unpack_batch_predict(batch, self.opt["cuda"])  # 1
        
        Item_feature = torch.index_select(self.item_hidden_out, 0, self.gen.embedding.item_index) # item_num * hidden_dim
        User_feature = torch.index_select(self.user_hidden_out, 0, User_One) # User_num * hidden_dim
        Global_repr = self.Graph_repr
        Global_repr = torch.index_select(Global_repr, 0, User_One)

        User_feature = User_feature.unsqueeze(1)
        User_feature = User_feature.repeat(1, self.opt["number_item"], 1) # (bs, 1650, 64)
        Global_repr = Global_repr.unsqueeze(1)
        Global_repr = Global_repr.repeat(1, self.opt["number_item"], 1)

        Item_feature = Item_feature.unsqueeze(0)
        Item_feature = Item_feature.repeat(User_feature.size()[0], 1, 1) 
        # Feature1 = torch.cat((User_feature, Item_feature),
        #                     dim=-1)
        Feature1 = Item_feature - User_feature
        output = self.gen.score_predict(Global_repr, Feature1)
        output_list, recommendation_list = output.sort(descending=True)
        return recommendation_list.cpu().numpy()
    
    def feature_corruption(self):
        user_index = torch.randperm(self.opt["number_user"], device=self.gen.embedding.user_index.device)
        item_index = torch.randperm(self.opt["number_item"], device=self.gen.embedding.user_index.device)
        user_feature = self.gen.embedding.user_embedding(user_index)
        item_feature = self.gen.embedding.item_embedding(item_index)
        return user_feature, item_feature

    def update_bipartite(self, UV_adj, VU_adj, adj,fake = 0):
        # We do not use any side information. if have side information, modify following codes.
        if fake:
            user_feature, item_feature = self.feature_corruption()
            user_feature = user_feature.detach()
            item_feature = item_feature.detach()
        else :
            user_feature = self.gen.embedding.user_embedding(self.gen.embedding.user_index)
            item_feature = self.gen.embedding.item_embedding(self.gen.embedding.item_index)

        self.user_hidden_out, self.item_hidden_out = self.gen.embedding(user_feature, item_feature, UV_adj, VU_adj, adj)

    
    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def reconstruct(self, UV, VU, UV_rated, VU_rated, relation_UV_adj, relation_VU_adj, adj,CUV, CVU,fake_adj, batch):
        self.gen.train()
        ### training discriminator
        self.dis.train()

        self.update_bipartite(CUV, CVU, fake_adj, fake = 1)
        fake_user_hidden_out = self.user_hidden_out
        fake_item_hidden_out = self.item_hidden_out

        self.update_bipartite(UV, VU, adj)
        user_hidden_out = self.user_hidden_out
        item_hidden_out = self.item_hidden_out
        
        if self.opt["number_user"] * self.opt["number_item"] > 10000000:
            user_One, item_One, neg_item_One, User_index_One, Item_index_One, real_user_index_id_Two, fake_user_index_id_Two, real_item_index_id_Two, fake_item_index_id_Two  = self.unpack_batch_DGI(batch, self.opt[
                "cuda"])
        else :
            user_One, item_One, neg_item_One = self.unpack_batch(batch, self.opt[
                "cuda"])

        mixup_real, mixup_fake, sub_global, graph_global = self.gen(user_hidden_out, item_hidden_out, fake_user_hidden_out, fake_item_hidden_out,
                                         UV, VU, CUV, CVU, user_One, item_One,
                                         UV_rated, VU_rated, relation_UV_adj, relation_VU_adj)
                                        
        self.Graph_repr = sub_global #(943, 64)

        dis_real = self.dis(mixup_real).view(-1)
        losssD_real = self.criterion(dis_real, torch.ones_like(dis_real))
        dis_fake = self.dis(mixup_fake).view(-1)
        lossD_fake = self.criterion(dis_fake, torch.zeros_like(dis_fake))
        lossD = (losssD_real + lossD_fake)/2
        self.optim_D.zero_grad()
        lossD.backward(retain_graph=True)
        self.optim_D.step()
        
        ### train generator

        user_feature_Two = self.my_index_select(user_hidden_out, user_One)
        sub_global_Two = self.my_index_select(sub_global, user_One)
        item_feature_Two = self.my_index_select(item_hidden_out, item_One)
        neg_item_feature_Two = self.my_index_select(item_hidden_out, neg_item_One)

        pos_One = self.gen.score(sub_global_Two, item_feature_Two - user_feature_Two)
        # pos_One = self.gen.score(sub_global_Two, item_feature_Two - user_feature_Two)
        # contrast_neg = self.gen.view(mixup_fake)
        # contrast_pos = self.gen.view(mixup_real)
        neg_One = self.gen.score(sub_global_Two, neg_item_feature_Two - user_feature_Two)
        # neg_One = self.gen.score(sub_global_Two, neg_item_feature_Two - user_feature_Two) 

        # pos_One = self.gen.score(mixup_real)
        # neg_One = self.gen.score(mixup_fake) 

        if self.opt["wiki"]:
            Label = torch.cat((torch.ones_like(pos_One), torch.zeros_like(neg_One))).cuda()
            pre = torch.cat((pos_One, neg_One))
            reconstruct_loss = self.criterion(pre, Label)
        else:
            # print('Rank Loss of contrast {}'.format(self.HingeLoss(contrast_pos, contrast_neg)))
            reconstruct_loss = self.HingeLoss(pos_One, neg_One)

        # mixup_real, mixup_fake = self.gen(self.user_hidden_out, self.item_hidden_out, fake_user_hidden_out,
        #                                 fake_item_hidden_out, UV, VU, CUV, CVU, user_One, item_One)                                      
        G_real = self.dis(mixup_real).view(-1)
        lossG_real = self.criterion(G_real, torch.ones_like(G_real))
        G_fake = self.dis(mixup_fake).view(-1)
        lossG_fake = self.criterion(G_fake, torch.zeros_like(G_fake))
        lossG = (lossG_real + lossG_fake)/2
        # lossG = lossG_real
        lossG2 = (1 - self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * lossG
        self.epoch_rec_loss.append((1 - self.opt["lambda"]) * reconstruct_loss.item())
        self.epoch_dgi_loss.append(self.opt["lambda"] * lossG.item())
        self.optim_G.zero_grad()
        lossG2.backward()
        self.optim_G.step()

        return lossG2.item()