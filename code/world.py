
import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "/Users/gus/Desktop/light-gcnII"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))

# if not os.path.exists(FILE_PATH):
#     os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'ml-1m','ml-100k']
all_models  = ['mf', 'p_gcn']
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['A_split'] = False
config['bigdata'] = False
config['alpha']= args.alpha
config['lambda_q'] = args.lambda_q
config['gamma'] = args.gamma
config['delta'] = args.delta
config['beta'] = args.beta

GPU = torch.cuda.is_available()
device = torch.device('cuda:'+str(args.gpu)+'' if GPU else "cpu")
# device1 = torch.device('cuda:1' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

id=args.graphID
dataset = args.dataset
model_name = args.model
save = args.save

a = eval(args.k)
K = a+[args.k_value]*(args.layer-len(a))

if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


TRAIN_epochs = args.epochs
stop = args.stop
topks = eval(args.topks)
t = eval(args.t)
beta = args.beta
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

