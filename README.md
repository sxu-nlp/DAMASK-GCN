## DAMASK_GCN-pytorch

## Introduction

We designed the DAMASK_GCN model,it contains three parts:  
1.We firstly construct multiple views.  
2.We use the **Personalized GCN with dilated mask convolution mechanism(P-GCN)** to learn the rating matrix between the user and item in every view.  
3.We use DAMASK_GCN to integrate multiple view of the rating matrix.




## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide three processed datasets: LastFM, Movielens-100k , Movielens-1M.

see more in `dataloader.py`

## An example to create multiple graphs
create two graphs on  **lastfm** dataset:
* change base directory

Change `ROOT_PATH` in `code/world.py`

* command  
` cd code && python create_graph.py --dataset="lastfm" --t="[64,52]" --beta="[1e-6,1e-5]"`

**note: if you want to run our model on your dataset , please create the multiple graphs firstly!**

## An example to run a 3-layer P-GCN on graph-1

run P-GCN on **lastfm** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command

` cd code && python main.py --decay=1e-3 --lr=0.001 --layer=3 --dataset="lastfm" --alpha=0.4 --graphID="1" --lambda_q=100 --beta=0.5 --delta=1 --save=0`

## An example to run DAMASK_GCN

run DAMASK_GCN on **ml-100k** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command
` cd code && python run.py --dataset="ml-100k" `

## the experimental parameters on three data sets
* lastfm  
alpha:  graph1=0.4, graph2=0.4, graph3=0.4  
decay=1e-3  
lr=0.001  
lambda_q=100 
beta=0.5 
delta=1 

* ml-100k  
alpha: graph1=0.4, graph2=0.4, graph3=0.4  
decay=1e-5  
lr=0.001  
lambda_q=50 
beta=0.65 
delta=1 

* ml-1m  
alpha: graph1=0.4, graph2=0.4, graph3=0.4  
decay=1e-4  
lr=0.001  
lambda_q=100 
beta=0.3 
delta=1.5 







