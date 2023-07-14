import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd

def getdata(filename):
    epoch = []
    precision = []
    recall = []
    ndcg = []

    with open(filename,'r') as file:
        while True:
            string = file.readline().strip('\n')
            if not string:break
            epoch.append(int(''.join(re.findall('(?<=epoch: )[0-9]+',string))))
            precision.append([float(i) for i in re.findall(r'\d+\.?\d*',''.join(re.findall(r'(?<=precision: \[)[0-9. ]+',string)))])#(?<=precision: \[)[0-9. ]+
            recall.append([float(i) for i in re.findall(r'\d+\.?\d*',''.join(re.findall(r'(?<=recall: \[)[0-9. ]+',string)))])
            ndcg.append([float(i) for i in re.findall(r'\d+\.?\d*',''.join(re.findall(r'(?<=ndcg: \[)[0-9. ]+',string)))])

    return epoch,np.array(precision),np.array(recall),np.array(ndcg)

def getMax(precision,recall,ndcg):
    max_precision = np.max(precision)
    max_recall = np.max(recall)
    max_ndcg = np.max(ndcg)
    print("max precision:",max_precision," max recall:",max_recall," max ndcg:",max_ndcg)
    # return max_precision,max_recall,max_ndcg

def show(epochs,precisions,recalls,ndcgs,lable=None,filepath=None,tag=[1,1,1],p=None):
    if p==None:
        p=list(i+1 for i in range(len(epochs)))
    if lable==None:
        lables = "light-gcn"
        k = len(epochs)
        if tag[0]==1:
            for i in range(k):
                plt.title('precision')
                plt.xlabel("epoch")
                plt.ylabel("precision")
                plt.plot(epochs[i], precisions[i], label=lables + str(p[i]))
                plt.legend()
            if filepath != None:
                plt.savefig(filepath+"precision.png")
            plt.show()

        if tag[1]==1:
            for i in range(k):
                plt.title('recall')
                plt.xlabel("epoch")
                plt.ylabel("recall")
                plt.plot(epochs[i], recalls[i], label=lables + str(p[i]))
                plt.legend()
            if filepath != None:
                plt.savefig(filepath+"recall.png")
            plt.show()

        if tag[2]==1:
            for i in range(k):
                plt.title('ndcg')
                plt.xlabel("epoch")
                plt.ylabel("ndcg")
                plt.plot(epochs[i], ndcgs[i], label=lables + str(p[i]))
                plt.legend()
            if filepath != None:
                plt.savefig(filepath+"ndcg.png")
            plt.show()
    else:
        lables = lable
        k = len(epochs)
        if tag[0]==1:
            for i in range(k):
                plt.title('precision')
                plt.xlabel("epoch")
                plt.ylabel("precision")
                plt.plot(epochs[i], precisions[i], label=lables[i])
                plt.legend()
            if filepath != None:
                plt.savefig(filepath+"precision.png")
            plt.show()

        if tag[1]==1:
            for i in range(k):
                plt.title('recall')
                plt.xlabel("epoch")
                plt.ylabel("recall")
                plt.plot(epochs[i], recalls[i], label=lables[i])
                plt.legend()
            if filepath != None:
                plt.savefig(filepath+"recall.png")
            plt.show()

        if tag[2]==1:
            for i in range(k):
                plt.title('ndcg')
                plt.xlabel("epoch")
                plt.ylabel("ndcg")
                plt.plot(epochs[i], ndcgs[i], label=lables[i])
                plt.legend()
            if filepath != None:
                plt.savefig(filepath+"ndcg.png")
            plt.show()

def showloss(epochs, loss, cos, labels):
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for i in range(len(epochs)):
        plt.plot(epochs[i][200:], loss[i][200:],label=str(labels[i]))
        plt.legend()
    plt.show()

    plt.title('Cos')
    plt.xlabel('epoch')
    plt.ylabel('cos')
    for i in range(len(epochs)):
        plt.plot(epochs[i][2:], cos[i][2:], label=str(labels[i]))
        plt.legend()
    plt.show()

def getloss(filename):
    epoch = []
    loss = []
    cos = []
    i = 0

    with open(filename, 'r') as file:
        while True:
            string = file.readline().strip('\n')
            if not string: break
            epoch.append(int(''.join(re.findall('(?<=epoch: )[0-9]+', string))))
            loss.append(float(''.join(re.findall('(?<=loss: )[0-9.]+', string))))
            cos.append(float(''.join(re.findall('(?<=cos: )[-0-9.]+', string))))
            i += 1

    return epoch, loss, cos

# 每一层多个模型的对比
def f1(layers,dataset,tag):
    for i in layers:
        path1 = "../result/" + dataset + "/" + str(i) + "layers/model1/"
        path2 = "../result/" + dataset + "/" + str(i) + "layers/model1/"
        # path3 = "../../DMV_GCN/result/" + dataset + "/" + str(i) + "layers/model1/"
        # path4 = "../../DMV_GCN/result/" + dataset + "/" + str(i) + "layers/model1/"

        label1 = "VS_LightGCN_1_17_3_" + str(i) + '_decay_0.005_lr_0.001.txt'
        label2 = "VS_LightGCN_1_23_1_1_" + str(i) + '_decay_0.005_lr_0.001.txt'
        # label3 = "VS_LightGCN_1" + str(i) + '_lr_0.001_decay_0.005.txt'
        # label4 = "VS_LightGCN" + str(i) + '_lr_0.001_decay_0.005.txt'
        # label3 = "LightGCN"+str(i)+"_7_0.005_0.001.txt"
        # label4 = "LightGCN" + str(i) + "_7_0.005_0.002.txt"

        file1 = path1 + label1
        file2 = path2 + label2
        # file3 = path3 + label3
        # file4 = path4 + label4
        # file3 = path+label3
        # file4 = path+label4
        epoch1, precision1, recall1, ndcg1 = getdata(file1)
        epoch2, precision2, recall2, ndcg2 = getdata(file2)
        # epoch3, precision3, recall3, ndcg3 = getdata(file3)
        # epoch4, precision4, recall4, ndcg4 = getdata(file4)
        epochs = [epoch1,epoch2]
        precisions = [precision1,precision2]
        recalls = [recall1,recall2]
        ndcgs = [ndcg1,ndcg2]

        lable = [label1,label2]
        show(epochs, precisions, recalls, ndcgs,lable=lable, filepath=path1, tag=tag)

#一个模型多层的变化
def f2(n,layers,dataset):
    epochs = []
    precisions = []
    recalls = []
    ndcgs = []

    for i in layers:
        filename = "../result/" + dataset + "/" + str(i) + "layers/model1/VS_LightGCN_1_17_3_"+str(i)+"_decay_0.005_lr_0.001.txt"
        epoch, precision, recall, ndcg = getdata(filename)
        epochs.append(epoch[n:])
        precisions.append(precision[n:])
        recalls.append(recall[n:])
        ndcgs.append(ndcg[n:])

    show(epochs, precisions, recalls, ndcgs, filepath="../result/" + dataset + "/", p=[i for i in layers])

def f3(filename,i):
    epoch, precision, recall, ndcg = getdata(filename)
    getMax(precision[:,i],recall[:,i],ndcg[:,i])

# 一个模型多层损失值和cos变化
def f4(dataset, layers,id, tag=''):
    file = "../result/" + dataset + '/'
    epochs = []
    loss = []
    cos = []
    labels = []
    for layer in layers:
        label = "LightGCN" + str(layer) + str(tag) + ".txt"
        filename = file + str(layer) + "layers/loss/" + label
        label = "DMV_GCN" + str(layer)
        epoch1, loss1, cos1 = getloss(filename)
        max_cos = np.min(np.array(cos1[200:]))
        min_loss = np.min(np.array(loss1))
        print(layer,"层：min cos: ",max_cos,"min loss: ",min_loss)
        epochs.append(epoch1)
        loss.append(loss1)
        cos.append(cos1)
        labels.append(label)
    showloss(epochs, loss, cos, labels)

def getTable(path):
    precision = pd.read_excel(path+"/precision.xlsx")
    recall = pd.read_excel(path + "/recall.xlsx")
    ndcg = pd.read_excel(path + "/ndcg.xlsx")
    label = list(precision.columns)[1:]
    return precision.to_numpy(),recall.to_numpy(),ndcg.to_numpy(),label

def showPlot(x,Y,title,xlabel,ylabel,path,tag):
    # maker_dic={0:"o",1:"^"}
    plt.plot(x,Y[0,1:],marker='o',label=Y[0,0],markersize=10)
    plt.plot(x, Y[-1, 1:], marker='^', label=Y[-1, 0],markersize=10)
    plt.title(title,fontsize=20)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel(ylabel,fontsize=15)
    plt.legend(prop={'family':'SimHei','size':15},loc=4)
    plt.savefig(path+'/'+str(dataset)+'_layer_'+tag+".jpg")
    plt.show()

def showBar(x,Y,title,xlabel,ylabel,path,tag):
    color_dic = {0: "darkorange", 1: "mediumaquamarine", 2: "cornflowerblue"}#mediumaquamarine
    xticks = np.arange(len(x))
    fig,ax = plt.subplots(figsize=(10,7))
    for i in range(len(Y)-1):
        ax.bar(np.array(xticks)+i*0.25,Y[i,1:],width=0.25,label=Y[i,0],color=color_dic[i])
    ax.set_title(title,fontsize=30)
    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_xticks(xticks+0.25)
    ax.set_xticklabels(x)
    ax.set_ylim(dic[dataset][tag]['ymin'],dic[dataset][tag]['ymax'])
    ax.legend(prop={'family':'SimHei','size':20},loc=1)
    plt.savefig(path+"/"+str(dataset)+'_'+tag+".jpg")
    plt.show()

if __name__=='__main__':
    dic = {"lastfm":{"Precision":{"ymin":0.064,"ymax":0.086},"Recall":{"ymin":0.23,"ymax":0.305},"NDCG":{"ymin":0.176,"ymax":0.235}},
           "ml-100k":{"Precision":{"ymin":0.05,"ymax":0.183},"Recall":{"ymin":0.2,"ymax":0.42},"NDCG":{"ymin":0.15,"ymax":0.35}},
           "ml-1m":{"Precision":{"ymin":0.185,"ymax":0.195},"Recall":{"ymin":0.292,"ymax":0.305},"NDCG":{"ymin":0.3,"ymax":0.326}}}
    dataset = "ml-1m"
    title="MoviLens-1M"
    path = "D:/刘峰/毕业/毕设/Recsys/实验数据/"+dataset
    precison,recall,ndcg,label = getTable(path)
    showBar(label,precison,title,"layer","Precision@20",path,"Precision")
    showBar(label, recall, title, "layer", "Recall@20", path,"Recall")
    showBar(label, ndcg, title, "layer", "NDCG@20", path,"NDCG")
    showPlot(label, precison, title, "layer", "Precision@20", path,"Precision")
    showPlot(label, recall, title, "layer", "Recall@20", path,"Recall")
    showPlot(label, ndcg, title, "layer", "NDCG@20", path,"NDCG")
    print()
    # # 每一层多个模型的对比
    # f1(layers=[48],dataset=dataset,tag=[1,1,1])

    # f2(0,[3,8,12,18,24,48,56],dataset)

    # for i in [3,8,12,18,24,48,56]:
    #     filename = "../result/" + dataset + "/" + str(i) + "layers/model1/VS_LightGCN_1_17_3_"+str(i)+"_decay_0.005_lr_0.001.txt"
    #     print("layer:", i)
    #     f3(filename,0)
    # n = 30
    # star = 1
    # end = 9
    # #一个模型多层的变化
    # f2(n,star,end,dataset)

    # epoch1,precision1,recall1,ndcg1 = getdata("../result1/"+dataset+"/"+str(7)+"layers/LightGCN"+str(7)+"(our).txt")
    # epoch2,precision2,recall2,ndcg2 = getdata("../result1/"+dataset+"/"+str(3)+"layers/LightGCN"+str(3)+".txt")
    #
    # epochs = [epoch1,epoch2]
    # precisions = [precision1,precision2]
    # recalls = [recall1,recall2]
    # ndcgs = [ndcg1,ndcg2]
    #
    # #显示lightGCN最好效果和our model最好效果
    # show(epochs,precisions,recalls,ndcgs,lable=["lightGCN"+str(7)+"(our).txt","lightGCN"+str(3)+".txt"])
    # star = 3
    # end = 5
    # j = 0
    # for i in range(star,end,1):
    #     print("layer:",i)
    #     f3(filename="../result/lastfm/"+str(i)+"layers/DMV_GCN.txt",i=j)
        # layers = [i]
        # f4(dataset, layers, tag="(alpha0.4dr)")

    # layers = [1,2,3,4,5,6,7,8,9]
    # f4(dataset, layers,id=3, tag="(alpha0.4dr)")#(alpha0.4dr)
