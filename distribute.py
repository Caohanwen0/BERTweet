from cmath import inf
import bmtrain as bmt
bmt.init_distributed(seed=0)
import torch
from model_center.model import Bert, BertConfig,Roberta,RobertaConfig
from model_center.layer import Linear
from model_center.dataset.bertdataset import DATASET
from model_center.dataset import DistributedDataLoader
from model_center.dataset import MMapIndexedDataset, DistributedMMapIndexedDataset, DistributedDataLoader
from model_center.tokenizer import BertTokenizer, RobertaTokenizer
from model_center.utils import print_inspect
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, Dataset
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
import pandas as pd
import json,os
from tqdm import tqdm
from arguments import get_args
from torch.utils.tensorboard import SummaryWriter

from prepare_dataset_Incivility import get_incivility_dataset
from prepare_dataset_SamVal import get_train_dataset, get_test_dataset
from prepare_dataset_3A import get_3A_dataset
from prepare_dataset_Dreaddit import get_Dreaddit_dataset
from prepare_dataset_AGNews import get_AGNews_dataset
from prepare_dataset_Article_Bias import get_Article_Bias_dataset
# log_iter = 50
# label_num = 2
# epochs = 30
# batch_size = 32
# learning_rate = 1e-5
# warm_up_ratio = 0.1

# bmt.print_rank("torch version", torch.__version__)
# bmt.print_rank(torch.cuda.get_arch_list())

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

args = initialize()
print(args)
if bmt.rank() == 0:
    writer = SummaryWriter(os.path.join(args.save, 'tensorborads'))

bmt.print_rank(f"Local rank:{bmt.rank()} | World size:{bmt.world_size()}")
temp_texts, temp_labels = get_train_dataset() # 这里的temp_texts, temp_labels 就是两个同样长度的list，前者是raw text，后者是label（0，1，2）
test_texts, test_labels = get_test_dataset() #test_texts, test_labels 同上
#temp_texts, temp_labels = get_incivility_dataset('data/Reddit_Incivility/incivility_coded_0302.train.json')
#test_texts, test_labels = get_incivility_dataset('data/Reddit_Incivility/incivility_coded_0302.test.json')
# temp_texts, temp_labels = get_3A_dataset('data/SemEval3A/train_3A.txt', do_preprocess = True)
# test_texts, test_labels = get_3A_dataset('data/SemEval3A/gold_test_3A.txt', do_preprocess =True)
# temp_texts, temp_labels = get_Dreaddit_dataset('data/Dreaddit/dreaddit-train.csv')
# test_texts, test_labels = get_Dreaddit_dataset('data/Dreaddit/dreaddit-test.csv')
#temp_texts, temp_labels, test_texts, test_labels= get_Article_Bias_dataset('data/Article-Bias-Prediction/jsons')
#temp_texts, temp_labels = get_AGNews_dataset('data/AGNews/train.csv')
#test_texts, test_labels = get_AGNews_dataset('data/AGNews/test.csv')


assert len(set(temp_labels))== len(set(test_labels))
label_num = len(set(test_labels))

class RobertaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if args.checkpoint is not None:
            bmt.print_rank(f"Initializing bert from pretrained {args.checkpoint}...")
            self.roberta = Roberta.from_pretrained(args.checkpoint) # load roberta from pretrained
        else: 
            assert args.load is not None
            bmt.print_rank(f"Initializing bert from our model path {args.load}...")
            self.roberta = Roberta(config)
            bmt.load(self.roberta, args.load)
        print_inspect(self.roberta, "*")
        self.dense = Linear(config.dim_model, label_num)
        bmt.init_parameters(self.dense) # init dense layer

    def reload(self, config):
        super().__init__()
        if args.checkpoint is not None:
            bmt.print_rank(f"Initializing bert from pretrained {args.checkpoint}...")
            self.roberta = Roberta.from_pretrained(args.checkpoint) # load roberta from pretrained
        else: 
            assert args.load is not None
            bmt.print_rank(f"Initializing bert from our model path {args.load}...")
            self.roberta = Roberta(config)
            bmt.load(self.roberta, args.load)
        print_inspect(self.roberta, "*")
        self.dense = Linear(config.dim_model, label_num)
        bmt.init_parameters(self.dense) # init dense layer 

    def forward(self, input_ids, attention_mask):
        pooler_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

if args.checkpoint is not None:
    config = RobertaConfig.from_pretrained(args.checkpoint)
else:
    config = RobertaConfig.from_json_file(args.model_config)

model = RobertaModel(config)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    temp_texts,
    temp_labels,
    random_state = 2020, 
    test_size=0.1,
    stratify = temp_labels) # split train and val
bmt.print_rank(f"Train size: {len(train_labels)} | Val size: {len(val_labels)} | Test size: {len(test_labels)}")
bmt.print_rank(f"{label_num} class classification...")

if args.checkpoint:
    tokenizer = RobertaTokenizer.from_pretrained(args.checkpoint)
else:
    tokenizer_obj = Tokenizer.from_file(args.tokenizer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.pad_token = '<pad>'
    tokenizer.eos_token = '</s>'
    tokenizer.sep_token = '<s>'
    tokenizer.unk_token = '<unk>'
    tokenizer.mask_token = '<mask>'

tokens_train = tokenizer.batch_encode_plus(
    train_texts,
    max_length = args.padding_len,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

tokens_val = tokenizer.batch_encode_plus(
    val_texts,
    max_length = args.padding_len,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

tokens_test = tokenizer.batch_encode_plus(
    test_texts,
    max_length = args.padding_len,
    padding='max_length',
    truncation=True,
    add_special_tokens = True,
)

train_data = TensorDataset(torch.tensor(tokens_train['input_ids']), \
    torch.tensor(tokens_train['attention_mask']), \
    torch.tensor(train_labels))
val_data = TensorDataset(torch.tensor(tokens_val['input_ids']), \
    torch.tensor(tokens_val['attention_mask']), \
    torch.tensor(val_labels))
test_data = TensorDataset(torch.tensor(tokens_test['input_ids']), \
    torch.tensor(tokens_test['attention_mask']), \
    torch.tensor(test_labels))

train_dataloader = DistributedDataLoader(train_data, batch_size = args.batch_size, shuffle = True)
val_dataloader = DistributedDataLoader(val_data, batch_size = args.batch_size, shuffle = False)
test_dataloader = DistributedDataLoader(test_data, batch_size = args.batch_size, shuffle = False)

# optimizer and lr-scheduler
total_step = (len(train_dataloader)) * args.epochs
optimizer = bmt.optim.AdamOptimizer(model.parameters(),lr = 1e-5, betas=(0.9,0.98))
# optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())
# lr_scheduler = bmt.lr_scheduler.Linear(
#     optimizer, 
#     start_lr = 1e-5,
#     warmup_iter = warm_up_ratio * total_step, 
#     end_iter = total_step)

loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    
def fine_tune():
    best_valid_acc = 0
    valid_acc_list = []
    early_stopping = 0
    global_step = 0
    for epoch in range(args.epochs):
        bmt.print_rank("Epoch {} begin...".format(epoch + 1))
        model.train()
        for step, data in enumerate(train_dataloader):
            global_step += 1
            input_ids, attention_mask, labels = data
            # load to cuda
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            global_loss = bmt.sum_loss(loss).item()
            if bmt.rank() == 0:
                writer.add_scalar(f"Loss/train", global_loss, global_step)
            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups,max_norm= float('inf'), scale = optimizer.scale, norm_type = 2)
            # bmt.optim_step(optimizer, lr_scheduler)
            bmt.optim_step(optimizer) # fixed learning rate
            if step % args.inspect_iters == 0:
                bmt.print_rank(
                    "loss: {:.4f} | scale: {:10.4f} | grad_norm: {:.4f} |".format(
                        global_loss,
                        int(optimizer.scale),
                        grad_norm,
                    )
                )
        model.eval()
        with torch.no_grad():
            pd = [] # prediction
            gt = [] # ground_truth
            for data in val_dataloader:
                input_ids, attention_mask, labels = data
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                logits = model(input_ids, attention_mask)
                loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
                logits = logits.argmax(dim=-1)
                pd.extend(logits.cpu().tolist())
                gt.extend(labels.cpu().tolist())

            # gather results from all distributed processes
            pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
            gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

            # calculate metric
            acc = accuracy_score(gt, pd)
            if bmt.rank()==0:
                writer.add_scalar(f"Acc/dev", acc, epoch)
            early_stopping += 1
            bmt.print_rank(f"validation accuracy: {acc*100:.2f}\n")
            if acc > best_valid_acc:
                best_valid_acc = acc
                bmt.print_rank("saving the new best model...\n") # save checkpoint
                bmt.save(model, os.path.join(args.save, 'model.pt'))
                early_stopping = 0 
            valid_acc_list.append(acc)
            # if early_stopping > 5:
            #     bmt.print_rank("Accuracy have not rising for 5 epochs.Early stopping..")
            #     break # break for iter

def check_performance():
    bmt.load(model, os.path.join(args.save, 'model.pt'))
    bmt.print_rank("Checking performance...\n")
    with torch.no_grad():
        pd = [] # prediction
        gt = [] # ground_truth
        for data in test_dataloader:
            input_ids, attention_mask, labels = data
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(labels.cpu().tolist())

        # gather results from all distributed processes
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        bmt.print_rank(classification_report(y_true = gt, y_pred = pd, digits = 5))

fine_tune()
check_performance()