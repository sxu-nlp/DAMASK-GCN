
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go DAMASK_GCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of DAMASK_GCN/P_GCN")
    parser.add_argument('--layer', type=int,default=18,
                        help="the layer num of DAMASK_GCN/P_GCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=0.001,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like ml-1m")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='lastfm',
                        help="available datasets: [lastfm,mk-100k,ml-1m]")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--comment', type=str,default="p_gcn")
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='p_gcn', help='rec-model, support [mf, p_gcn]')
    parser.add_argument('--graphID', type=str, default="1",help="graph1 graph2 graph3,support [1,2,3]")
    parser.add_argument('--gamma', type=float, default=0.4, help="the Personalized represent weight,supoort [0-1]")
    parser.add_argument('--save',type=int,default=0,help='1 : save the embedding')
    parser.add_argument('--stop',type=int,default=10,
                        help='Select the epoch with the highest performance in P_GCN for DAMASK_GCN')
    parser.add_argument('--t',nargs='?',default="[64,52]",
                        help='t concepts for constructing multiple views(view1,view2),t support [1-64]')
    parser.add_argument('--alpha',nargs='?',default="[1e-6,1e-5]",
                        help='alpha for filtering the similarity of the lower edge. ')
    parser.add_argument('--k', default="[100,80,50]")
    parser.add_argument('--k_value', default=10)
    parser.add_argument('--lambda_q', type=int, default=100)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--gpu',type=int,default=0)
    return parser.parse_args()
