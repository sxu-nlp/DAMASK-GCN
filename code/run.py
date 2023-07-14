import world
import Procedure
import model
import dataloader
import os
import time
from world import cprint
import utils
import torch.nn.functional as F
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

if __name__=='__main__':
    world.dataset = 'lastfm'
    if world.dataset == 'lastfm':
        dataset = dataloader.LastFM(1)
    elif world.dataset in ['ml-100k', 'ml-1m']:
        dataset = dataloader.Movielens(1, path="../data/" + world.dataset)
    world.config['n_layers'] = 3
    world.config['lr'] = 0.001
    Model = model.DAMASK_GCN(world.config,dataset)
    Model = Model.to(world.device)
    bpr = utils.DAMASK_GCNLoss(Model, world.config)

    filepath = "../result/" + str(world.dataset) + '/' + str(world.config['n_layers']) + "layers"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = filepath + "/DMV_GCN.txt"

    start = time.time()
    best = {'recall':0,'precision':0,'ndcg':0}
    best_id = 0
    step = 0
    stop = 5
    for epoch in range(world.TRAIN_epochs):
        if epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Model, filename,epoch, world.config['multicore'])
            if results['ndcg']>best['ndcg']:
                step=0
                best = results
                best_id = epoch
            else:
                step+=1

        output_information = Procedure.DMV_train(dataset, Model, bpr)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')

        if step>stop:
            end = time.time()
            print("time:",end-start)
            print("best epoch:",best_id," ",best)
            exit(0)
