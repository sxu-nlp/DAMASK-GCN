
import world
import numpy as np
import torch
import utils
from utils import timer
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class,epoch,lossfile):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class
    # bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    loss_w_all = 0.
    # cos = Recmodel.getCosin()
    for (batch_i, (batch_users, batch_pos, batch_neg)) in \
            enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        cri,loss_w = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        loss_w_all+=loss_w
    aver_loss = aver_loss / total_batch
    loss_w_all = loss_w_all/total_batch
    time_info = timer.dict()
    timer.zero()
    with open(lossfile, 'a') as file:
        file.writelines("epoch: "+str(epoch)+" loss: "+str(aver_loss)+" loss1: "+str(loss_w_all)+'\n')
    return f"loss{aver_loss:.3f}-loss_w{loss_w_all:.6f}-{time_info}"

def BPR_train_original1(dataset, recommend_model, loss_class1,epoch,lossfile):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class1
    # bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    loss_w_all = 0
    w_all = np.array(Recmodel.attention_weight.tolist())
    # cos = Recmodel.getCosin()
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        cri,w_loss = bpr.stageOne1(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        loss_w_all += w_loss
    aver_loss = aver_loss / total_batch
    loss_w_all = loss_w_all/total_batch
    time_info = timer.dict()
    timer.zero()
    with open(lossfile, 'a') as file:
        file.writelines("epoch: "+str(epoch)+" loss: "+str(aver_loss)+" loss1: "+str(loss_w_all)+"weight:"+str(w_all)+'\n')
    return f"loss{aver_loss:.3f}-weight{w_all}-loss_w{loss_w_all:.6f}-{time_info}"

def BPR_train_original2(dataset, recommend_model, loss_class1,weight_opt,epoch,lossfile):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class1
    # bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    loss_w_all = 0
    w_all = 0
    loss_w_all = world.config['lambda_q']*Recmodel.getWeightLoss()
    w_all = np.array(Recmodel.attention_weight.tolist())
    weight_opt.zero_grad()
    loss_w_all.backward()
    weight_opt.step()
    # cos = Recmodel.getCosin()
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    with open(lossfile, 'a') as file:
        file.writelines("epoch: "+str(epoch)+" loss: "+str(aver_loss)+" loss1: "+str(loss_w_all)+"weight:"+str(w_all)+'\n')
    return f"loss{aver_loss:.3f}-weight{w_all}-loss_w{loss_w_all:.6f}-{time_info}"

def DMV_train(dataset, recommend_model, loss_class):
    Recmodel = recommend_model
    Recmodel.train()
    bpr = loss_class
    # bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    al_weight_reg = 0
    a_all = 0
    for (batch_i, (batch_users, batch_pos, batch_neg)) in \
            enumerate(utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):
        cri,a,weight_reg = bpr.stageOne(batch_users, batch_pos)
        aver_loss += cri
        al_weight_reg+=weight_reg
        a_all+=a
    aver_loss = aver_loss / total_batch
    al_weight_reg = al_weight_reg/total_batch
    a_all = a_all/total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-a{a_all:.6f}-weight_reg{al_weight_reg:.6f}-{time_info}"

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def Test(dataset, Recmodel,filename,epoch=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if multicore == 1:
            pool.close()

        if epoch==None:
            with open(filename, 'a') as file:
                file.writelines("DMV_GCN result precision: " + str(results['precision'])
                                + " recall: " + str(results['recall']) + " ndcg: " + str(results['ndcg']) + '\n')
        else:
            with open(filename, 'a') as file:
                file.writelines("epoch: " + str(epoch) + " precision: " + str(results['precision'])
                                + " recall: " + str(results['recall']) + " ndcg: " + str(results['ndcg']) + '\n')
        print(results)
        return results
