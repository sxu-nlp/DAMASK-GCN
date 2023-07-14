import dataloader
import torch
import numpy as np
import copy
from sklearn.metrics.pairwise import cosine_similarity

#计算两个矩阵之间的余弦相似度
#Cos(A,B) = (A*B)/(||A||*||B||)
def Cos(A,B):
    norm_A = torch.norm(A,2,-1,keepdim=True)
    norm_B = torch.norm(B,2,-1,keepdim=True)

    A_norm = A/norm_A
    B_norm = B/norm_B

    cos = torch.matmul(A_norm,B_norm.T)
    return cos

def creatNeighberMetrix(x,n=None):
    lis1 = x
    lis2 = [(b,a) for a,b in lis1]
    if n!=None:
        lis = lis1+lis2+[(i,i) for i in range(n)]
    else:
        lis = lis1+lis2

    A = np.zeros(shape=(11,11))

    for a,b in lis:
        A[a-1,b-1] = 1

    return A

def LaplasNorm(A):
    D = A.sum(axis=0)
    D_inv = np.power(D,-0.5)
    D_inv[np.isnan(D_inv)] = 0
    D_mat = np.diag(D_inv)

    L = np.matmul(D_mat,A)
    L = np.matmul(L,D_mat)

    return L

def comput_lgc(L,n):
    num = len(L)
    E = np.identity(num)
    A = L
    A_i = L
    for i in range(1,n):
        A_i = np.matmul(A_i,L)
        A += A_i
    A = E+A

    return A/(n+1)

def comput_dmv(L,n):
    num = len(L)
    E = np.identity(num)
    A = n*L
    A_i = L
    for i in range(1, n):
        A_i = np.matmul(A_i, L)
        A += (n-i)*A_i
    A = E + A

    return A / (n + 1)

def comput_A_n(A,n):
    A_i = A
    for i in range(1,n):
        A_i = np.matmul(A_i,A)

    return A_i

def DMV_GCN_Q_i(K):
    Q_i = [K-i for i in range(K)]
    Q_i.insert(0,1)
    Q_i = np.array(Q_i)
    Q_i_len = Q_i/len(Q_i)
    Q_i_rate = Q_i/Q_i.sum()
    return Q_i_rate,Q_i_len,Q_i_len.sum()

def f1(delta,beta,k):
    Q_i = [(1 - delta) * beta * pow(1 - beta, i) for i in range(k)]
    Q_i.insert(0, delta)
    Q_i = np.array(Q_i)
    Q_i = Q_i / Q_i.sum()
    return Q_i

def f2(delta,beta,k):
    Q_i = [beta * pow(1 - beta, i) for i in range(k + 1)]
    Q_i = np.array(Q_i)
    Q_i_len = Q_i*delta
    Q_i_sum = Q_i / Q_i.sum()
    return Q_i_sum,Q_i_len

if __name__=="__main__":
    layer=18
    Q_i_rate,Q_i,Q_i_sum = DMV_GCN_Q_i(layer)
    print("DMV-GCN原始方式比例：",Q_i_rate)
    print("DMV-GCN原始方式：",Q_i)
    print("DMV-GCN和：",Q_i_sum)
    # Q1 = f1(0.01,0.3,layer)
    # print("方式1：",Q1)
    Q2_sum,Q2 = f2(Q_i_sum,0.1,layer)
    print("方式2比例：",Q2_sum)
    print("方式2：",Q2)
    # matrix = torch.rand(size=[3,2])
    # index_c,index_l = torch.where(matrix>0)
    # print(index_c)
    # # dataset = dataloader.LastFM()
    # # A = dataset.getSparseGraph()
    # # A_all = A.to_dense()
    # # cos = Cos(A_all,A_all)
    # # list_A = [A_all]
    # # #单独每一阶邻居的邻接矩阵
    # # for i in range(2,10):
    # #     A_all = torch.sparse.mm(A,A_all)
    # #     list_A.append(A_all)
    # # print(A.size())
    #
    # A = creatNeighberMetrix([(1,2),(1,6),(1,9),(1,3),(2,4),(2,5),(6,7),(6,8),(9,10),(3,11)])
    # L = LaplasNorm(A)
    # # y = comput_lgc(L,3)
    # y = comput_dmv(L,3)
    # # y = comput_A_n(L,3)
    # print("y:\n",y)
    #
    #
    # a = y[0,:]
    # b = y[1,:]
    # # a = np.array([1,1,1,0,0,1,0,0,1,0,0])
    # # b = np.array([1,1,0,1,1,0,0,0,0,0,0])
    # # b = np.ones(shape=(11,))
    #
    # print("the cos betweein in a and b is: ",cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))[0][0])