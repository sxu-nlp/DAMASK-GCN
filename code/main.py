import world
import utils
from world import cprint
import time
import Procedure
import os
from torch import optim

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

if __name__ == '__main__':
    id = world.id
    Recmodel = register.MODELS[world.model_name](world.config, dataset, id)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    param = list(Recmodel.parameters())
    weight_opt = optim.Adam([{'params': param[0], 'lr': 1e-3},{'params': param[1], 'lr': 1e-3}])

    model_name = "/DAMASK-GCN_beta_"+str(world.config['delta'])+"_beta_"+str(world.config['beta'])+"_"

    filepath = "../result/" + str(world.dataset) +'/'+ str(Recmodel.n_layers) + "layers"+ '/model'+str(id)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = filepath + model_name + str(Recmodel.n_layers)+"_decay_"+str(world.config['decay'])+"_lr_"+str(world.config['lr'])+".txt"

    losspath = filepath + '/loss'
    if not os.path.exists(losspath):
        os.makedirs(losspath)
    lossfile = losspath + model_name + str(Recmodel.n_layers)+"_decay_"+str(world.config['decay'])+"_lr_"+str(world.config['lr'])+".txt"

    save = world.save
    best_result = {"precision":0,"recall":0,"ndcg":0}
    best_epoch = 0
    stop = world.stop
    step = 0
    start = time.time()
    Recmodel.getMaskAdj()
    print("create the adj time:",time.time()-start)
    start = time.time()
    Recmodel.computAttention()
    print("attention time:",time.time()-start)
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset, Recmodel, filename,epoch, world.config['multicore'])
            if result["ndcg"]>best_result["ndcg"]:
                best_result = result
                best_epoch = epoch
                step=0
            else:
                step+=1

            if epoch>300 and step >= stop:
                break
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, lossfile)
        end = time.time()
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} time:{end-start}')
    print("the best result epoch:",best_epoch," result:",best_result)

    if save == 1:
        Recmodel.saveEmbedding('../data/' + str(world.dataset) + '/model' + str(id) + "/layer" + str(Recmodel.n_layers))

    with open(filename, 'a') as file:
        file.writelines("the best result epoch: " + str(best_epoch) + " precision: " + str(best_result['precision'])
                        + " recall: " + str(best_result['recall']) + " ndcg: " + str(best_result['ndcg']) + '\n')
    exit(0)
