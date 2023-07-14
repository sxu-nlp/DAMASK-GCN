
import pandas as pd
import numpy as np
import os
import dataloader
import pickle
import world
from scipy.sparse import csr_matrix


def saveSVDEmbedding(dataset, user_file, item_file):
    '''
    generate the svd embedding\n
    .
    '''
    R = dataset.getUserItemNet().to_dense().cpu().numpy()
    u, sigma, vt = np.linalg.svd(R)
    v = vt.T
    user_embedding = u[:, :64]
    item_embedding = v[:, :64]

    with open(user_file, 'wb') as file_user:
        pickle.dump(user_embedding, file_user)
    with open(item_file, 'wb') as file_item:
        pickle.dump(item_embedding, file_item)


def getEmbedding(user_file, item_file):
    with open(user_file, 'rb') as file1:
        user_embedding = pickle.load(file1)
    with open(item_file, 'rb') as file2:
        item_embedding = pickle.load(file2)
    return user_embedding, item_embedding


def saveSVDGraph(user_embedding, item_embedding, a,filename1,filename2):
    '''
    create the multiple-view graphs\n
    .
    '''
    r = np.matmul(user_embedding, item_embedding.T)
    r[r < a] = 0

    data = pd.read_table(filename1, header=None,sep=' ').astype(int)
    user = np.array(data[:][0])
    n_users = np.max(user) + 1
    item = np.array(data[:][1])
    m_items = np.max(item) + 1
    UserItemNet = csr_matrix((np.ones(len(user)), (user, item)), shape=(n_users, m_items)).A

    mask_adj = UserItemNet * r
    user, item = np.where(mask_adj> 0)
    with open(filename2, "w") as file:
        for i in range(len(user)):
            file.write(str(user[i]) + " " + str(item[i]) + "\n")


if __name__ == "__main__":
    path = "../data/"
    data = world.dataset
    dataset_path = path+data
    model = ['model1','model2','model3']
    model_paths = [dataset_path + '/' + model[0],dataset_path+'/'+model[1],dataset_path+'/'+model[2]]
    user_files = [model_paths[0] + "/user_embedding.pkl",model_paths[1] + "/user_embedding.pkl",model_paths[2] + "/user_embedding.pkl"]
    item_files = [model_paths[0] + "/item_embedding.pkl",model_paths[1] + "/item_embedding.pkl",model_paths[2] + "/item_embedding.pkl"]
    graph_paths = [dataset_path+'/train.txt',model_paths[1]+'/graph.txt',model_paths[2]+'/graph.txt']

    for model_path in model_paths:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    if data == 'lastfm':
        dataset = dataloader.LastFM(1)
    elif data in ['ml-100k', 'ml-1m']:
        dataset = dataloader.Movielens(1, path="../data/" + data)

    #get the original graph svd embedding
    if not os.path.exists(user_files[0]):
        saveSVDEmbedding(dataset, user_files[0], item_files[0])
    user_embedding, item_embedding = getEmbedding(user_files[0], item_files[0])

    t1 = world.t[0]
    t2 = world.t[1]
    #generate the two graphs
    saveSVDGraph(user_embedding[:,:t1], item_embedding[:,:t1], world.beta[0],graph_paths[0],graph_paths[1])
    saveSVDGraph(user_embedding[:,:t2], item_embedding[:,:t2], world.beta[1],graph_paths[0],graph_paths[2])

    if data == 'lastfm':
        dataset = dataloader.LastFM(2)
    elif data in ['ml-100k', 'ml-1m']:
        dataset = dataloader.Movielens(2, path="../data/" + data)
    saveSVDEmbedding(dataset, user_files[1], item_files[1])

    if data == 'lastfm':
        dataset = dataloader.LastFM(3)
    elif data in ['ml-100k', 'ml-1m']:
        dataset = dataloader.Movielens(3, path="../data/" + data)
    saveSVDEmbedding(dataset, user_files[2], item_files[2])

    world.cprint('create the multiple graphs successfully!')
