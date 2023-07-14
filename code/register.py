
import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['ml-100k','ml-1m']:
    dataset = dataloader.Movielens(world.id,path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM(world.id)

print('===========config================')
pprint(world.config)
print("model id:",world.id)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'p_gcn': model.P_GCN
}

