
import world
import torch
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import pickle
from world import cprint
import numpy as np
import copy


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

class P_GCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset, i):
        super(P_GCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.id = i
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.gamma = self.config['gamma']
        self.indexs = torch.arange(0, self.num_items + self.num_users).reshape(-1, 1)
        self.A_0 = torch.eye(self.num_users + self.num_items).to(world.device)
        self.beta = self.config['beta']
        self.delta = self.config['delta']
        self.Q_i = self.getQ_i()
        self.X = self.getX()

        # generate the query vector
        self.vector_u = torch.nn.Parameter(torch.rand(size=(1, self.num_users + self.num_items)).to(world.device))
        self.vector_v = torch.nn.Parameter(torch.rand(size=(1, self.num_users + self.num_items)).to(world.device))

        self.k = world.K

        # load the svd embedding
        user_file = "../data/" + world.dataset + "/model" + str(self.id) + "/user_embedding.pkl"
        item_file = "../data/" + world.dataset + "/model" + str(self.id) + "/item_embedding.pkl"
        try:
            user_embdding, item_embedding = self.readEmbedding(user_file, item_file)
            self.user_emb0 = torch.tensor(user_embdding).to(world.device)
            self.item_emb0 = torch.tensor(item_embedding).to(world.device)
            self.embedding_user.weight.data = torch.tensor(user_embdding).to(world.device)
            self.embedding_item.weight.data = torch.tensor(item_embedding).to(world.device)
            world.cprint("load the svd embedding")
        except:
            world.cprint('please generate the svd embedding in create_graph.py')
            exit(0)

        self.f = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)
        self.crossEntropy = nn.CrossEntropyLoss()
        self.Graph, self.adj_mat = self.dataset.getSparseGraph()
        print(f"vs_lgn is already to go(dropout:{self.config['dropout']})")

    def getQ_i(self):
        Q_i = [self.beta*pow(1-self.beta,i) for i in range(self.n_layers+1)]
        Q_i = np.array(Q_i)
        Q_i = torch.tensor(Q_i,dtype=torch.float32).to(world.device)
        Q_i = Q_i / Q_i.sum()
        return Q_i

    def getX(self):
        X = self.delta
        return X

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def readEmbedding(self, user_file, item_file):
        '''
        load the pretrain svd embedding
        '''
        with open(user_file, 'rb') as file1:
            user_embedding = pickle.load(file1)
        with open(item_file, 'rb') as file2:
            item_embedding = pickle.load(file2)
        return user_embedding, item_embedding

    def getAttentionScore(self,quary):
        w_k = torch.mul(self.W,quary).sum()
        return w_k.flatten()

    def computer(self):
        """
        propagate methods for VS_LightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        all_emb0 = torch.cat([self.user_emb0, self.item_emb0])

        attention_weight = self.computAttention()

        embs = [self.X*attention_weight[0]*all_emb]
        emb0 = all_emb

        # 计算注意力分数
        for layer in range(self.n_layers):
            A_k = self.A_list[layer]
            all_emb = torch.mm(A_k, emb0) + self.gamma * all_emb0
            embs.append(self.X*attention_weight[layer+1]*all_emb)

        loss_w = F.mse_loss(attention_weight, self.Q_i)
        embs = torch.stack(embs, dim=1)
        light_out = torch.sum(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items,loss_w

    def computAttention(self):
        '''
        compute the new adj attention scores
        '''
        w_list = []
        # 计算注意力分数
        W_k_l = torch.matmul(self.vector_u, self.A_0)
        W_k = torch.matmul(W_k_l, self.vector_v.T).flatten()
        w_list.append(W_k)

        g_droped = self.Graph.to_dense()

        for layer in range(self.n_layers):
            A_k = g_droped

            # 计算注意力分数
            W_k_l = torch.mm(A_k, self.vector_v.T)
            W_k = torch.matmul(self.vector_u, W_k_l).flatten()
            w_list.append(W_k)

            g_droped = torch.sparse.mm(self.Graph,A_k)

        w_list = torch.stack(w_list, dim=1).flatten()
        w_list = w_list / w_list.sum()
        attention_weight = self.softmax(w_list)
        return attention_weight

    def getWeightLoss(self):
        attention_weight = self.computAttention()
        loss_w = F.mse_loss(attention_weight, self.Q_i)
        return loss_w

    def getMaskAdj(self):
        '''
        create the top n adj metrix
        '''
        self.A_list = []
        g_droped = self.Graph.to_dense()

        for layer in range(self.n_layers):
            A_k = g_droped
            if self.k[layer] != 0:
                values, index = torch.topk(A_k, k=int(self.k[layer]))
                indexs = copy.deepcopy(self.indexs).to(world.device)
                first_sub = indexs.expand(self.num_items + self.num_users, int(self.k[layer])).flatten().reshape(1, -1)
                secend_sub = index.reshape(1, -1)
                indexs = torch.cat([first_sub, secend_sub], dim=0)
                values = g_droped[first_sub, secend_sub]
                values = values.flatten()
                A_k = torch.sparse.FloatTensor(indexs, values, torch.Size(
                    [self.num_users + self.num_items, self.num_users + self.num_items]))
                self.A_list.append(A_k.to(world.device))
            else:
                self.A_list.append(g_droped)
            # self.A_list.append(A_k.to_sparse())
            g_droped = torch.sparse.mm(self.Graph, g_droped)

    def getUsersRating(self, users):
        all_users, all_items,_ = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items,weight_loss = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego,weight_loss

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0,weight_loss) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss,weight_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items,_ = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def saveEmbedding(self, filepath):
        '''
        save the embedding for DMV_GCN
        '''
        users, items,_ = self.computer()
        users = users.detach().cpu()
        items = items.detach().cpu()
        user_file = filepath + '/model_user_embedding.pkl'
        item_file = filepath + '/model_item_embedding.pkl'
        with open(user_file, 'wb') as file:
            pickle.dump(users, file)
        with open(item_file, 'wb') as file:
            pickle.dump(items, file)
        print('write over!')

    def getCosin(self):
        # norm = torch.norm(embedding,p=2,dim=1,keepdim=True)
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        embedding = torch.cat([users_emb, items_emb])
        adj = self.Graph.to_dense()
        cos_1 = embedding / (torch.norm(embedding, 2, -1, keepdim=True).expand_as(embedding) + 1e-12)
        cos = torch.matmul(cos_1,cos_1.T)
        cos_mask = cos*adj
        cos_mask = torch.sum(cos_mask)/(torch.count_nonzero(adj))
        return cos_mask.cpu().item()


class DAMASK_GCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(DAMASK_GCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.f = nn.Sigmoid()
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.n_layers = self.config['n_layers']
        try:
            # import the trained users embedding and items embedding
            self.user1_embedding, self.item1_embedding = self.getEmbedding('../data/' + str(world.dataset) + '/model1/layer' + str(self.n_layers))
            self.user2_embedding, self.item2_embedding = self.getEmbedding('../data/' + str(world.dataset) + '/model2/layer' + str(self.n_layers))
            self.user3_embedding, self.item3_embedding = self.getEmbedding('../data/' + str(world.dataset) + '/model3/layer' + str(self.n_layers))
            self.rating1 = torch.matmul(self.user1_embedding, self.item1_embedding.T).flatten().reshape(-1, 1)
            self.rating2 = torch.matmul(self.user2_embedding, self.item2_embedding.T).flatten().reshape(-1, 1)
            self.rating3 = torch.matmul(self.user3_embedding, self.item3_embedding.T).flatten().reshape(-1, 1)
            self.ratings = torch.cat([self.rating1, self.rating2, self.rating3], dim=1)
            self.dim = len(self.user1_embedding[0])
        except:
            cprint("please use VS_lightgcn to generate the user and item embeddings in main.py")
            exit(0)
        # get the indicative vectors from each view
        self.indicate_vector1 = self.getIndicate_Vector(self.user1_embedding, self.item1_embedding)
        self.indicate_vector2 = self.getIndicate_Vector(self.user2_embedding, self.item2_embedding)
        self.indicate_vector3 = self.getIndicate_Vector(self.user3_embedding, self.item3_embedding)

        self.h = nn.Parameter(torch.randn(2 * self.dim).to(world.device))
        self.Graph, self.adj_mat = self.dataset.getSparseGraph()

    def getEmbedding(self,filepath):
        '''
        read the pretrain embedding from VS_LightGCN
        '''
        user_file = filepath+'/model_user_embedding.pkl'
        item_file = filepath+'/model_item_embedding.pkl'

        with open(user_file,'rb') as file:
            user_embedding = pickle.load(file).to(world.device)
        with open(item_file,'rb') as file:
            item_embedding = pickle.load(file)
        return user_embedding,item_embedding.to(world.device)

    def getIndicate_Vector(self,user_embedding,item_embedding):
        '''
        generate the indicative vector
        '''
        user_embedding = user_embedding.expand(self.num_items,self.num_users,self.dim)
        user_embedding = user_embedding.permute(1,0,2)
        item_embedding = item_embedding.expand(self.num_users,self.num_items,self.dim)
        indicate_vector = torch.cat([user_embedding,item_embedding],dim=2)
        return indicate_vector

    def computer(self):
        '''
        attention fusion
        '''
        attention1 = torch.matmul(self.indicate_vector1,self.h)
        attention2 = torch.matmul(self.indicate_vector2,self.h)
        attention3 = torch.matmul(self.indicate_vector3,self.h)

        attention1 = attention1.flatten().reshape(-1,1)
        attention2 = attention2.flatten().reshape(-1,1)
        attention3 = attention3.flatten().reshape(-1,1)

        attention = torch.cat([attention1, attention2, attention3], dim=1)
        weight = F.softmax(attention, dim=1)
        rating = self.f(torch.sum(self.ratings * weight, dim=1))
        return rating

    def bpr_loss(self,user,pos,neg):
        ratings, weight_reg = self.computer()
        ratings = ratings.view(self.num_users,self.num_items)
        pos_scores = ratings[user,pos]
        neg_scores = ratings[user,neg]
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss,weight_reg

    def loss(self,user,item):
        '''
        caculate the loss value
        '''
        ratings,weight_reg = self.computer()
        ratings = ratings.view(self.num_users,self.num_items)
        rating = ratings[user,item]
        loss = torch.sum(-torch.log(rating))
        return loss,weight_reg

    def getUsersRating(self, users):
        rating,_ = self.computer1()
        # rating = self.rating
        item = torch.tensor([i for i in range(self.num_items)],dtype=int).to(world.device)
        ratings = []
        for user in users:
            index = torch.ones(size=[self.num_items],dtype=int).to(world.device) * user * self.num_items + item
            ratings.append(rating[index.tolist()].tolist())
        ratings = torch.tensor(ratings).view(len(users), self.num_items)
        ratings = self.f(ratings)
        return ratings

