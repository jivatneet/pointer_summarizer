import os

root_dir = os.path.expanduser("~")

train_data_path = os.path.join(root_dir, "/raid/banerjee/lcq1-3253.500dim.shuffled.txt")
#train_data_path = os.path.join(root_dir, "pointer_summarizer/data/intermediatesparqls/lcquad2sparqltrainintermed1.json")
eval_data_path = os.path.join(root_dir, "/raid/kaur/val_lcq1_200_gold5.txt")
#decode_data_path = os.path.join(root_dir, "/raid/kaur/test_lcq1_1000_gold5.txt")
decode_data_path = os.path.join(root_dir, "/raid/kaur/lcq1traingold4.txt")
vocab_path = os.path.join(root_dir, "workspace-lcq3253/pointer_summarizer/lcq1vocab.txt")
log_root = os.path.join(root_dir, "/raid/kaur/pointer_summarizer2/log")

# Hyperparameters
hidden_dim= 256
emb_dim= 500
batch_size= 8
max_enc_steps=400
max_dec_steps=100
beam_size=10
min_dec_steps=0
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
ls_eps = 0.1 # label smoothing
max_iterations = 1000000
max_epochs = 5000

use_gpu=True

lr_coverage=0.15
use_lstm=True
dataset_size = 3800 
