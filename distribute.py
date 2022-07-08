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
import pandas as pd
import json
from tqdm import tqdm

from preprocess.prepare_dataset_SamVal import get_train_dataset, get_test_dataset
from preprocess.prepare_dataset_3A import get_3A_dataset
ROBERTA_MODEL = 'roberta-base'
PADDING_LEN = 512

log_iter = 50
label_num = 2
epochs = 30
batch_size = 32
learning_rate = 1e-5
warm_up_ratio = 0.1

PATH_TO_DATASET = 'data'
BERT_PATH = '/data0/private/caohanwen/OpenSoCo/checkpoint/reddit_twitter/checkpoint-46499.pt'
CHECKPOINT_PATH = 'saved_models/model.pt'
SAVED_PATH = 'saved_models/valid_acc.csv'

# bmt.print_rank("torch version", torch.__version__)
# bmt.print_rank(torch.cuda.get_arch_list())

class RobertaModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.roberta = Roberta.from_pretrained(ROBERTA_MODEL) # load roberta from pretrained
        self.roberta = Roberta(config)
        bmt.load(self.roberta, BERT_PATH)
        # print_inspect(self.roberta, "*")
        self.dense = Linear(config.dim_model, label_num)
        bmt.init_parameters(self.dense) # init dense layer

    def forward(self, input_ids, attention_mask):
        pooler_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = self.dense(pooler_output)
        return x


config = RobertaConfig.from_pretrained(ROBERTA_MODEL)
model = RobertaModel(config)

bmt.print_rank("Prepare dataset...")
bmt.print_rank(f"local rank:{bmt.rank()}, world size:{bmt.world_size()}")
# temp_texts, temp_labels = get_train_dataset() # 这里的temp_texts, temp_labels 就是两个同样长度的list，前者是raw text，后者是label（0，1，2）
# test_texts, test_labels = get_test_dataset() #test_texts, test_labels 同上

temp_texts, temp_labels = get_3A_dataset('data/SemEval3A/train_3A.txt', do_preprocess = True)
test_texts, test_labels = get_3A_dataset('data/SemEval3A/gold_test_3A.txt', do_preprocess =True)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    temp_texts,
    temp_labels,
    random_state = 2022, 
    test_size=0.1, 
    stratify = temp_labels
) # split train and val
bmt.print_rank(f"train, val and test size is {len(train_labels)}, {len(val_labels)}, {len(test_labels)}")
# tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


tokenizer_obj = Tokenizer.from_file('/root/bm_train_codes/tokenizer/tokenizer_.json')
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
tokenizer.pad_token = '<pad>'
tokenizer.eos_token = '</s>'
tokenizer.sep_token = '<s>'
tokenizer.mask_token = '<mask>'

tokens_train = tokenizer.batch_encode_plus(
    train_texts,
    max_length = PADDING_LEN,
    padding='max_length',
    truncation=True
)

tokens_val = tokenizer.batch_encode_plus(
    val_texts,
    max_length = PADDING_LEN,
    padding='max_length',
    truncation=True
)

tokens_test = tokenizer.batch_encode_plus(
    test_texts,
    max_length = PADDING_LEN,
    padding='max_length',
    truncation=True
)

train_data = TensorDataset(torch.tensor(tokens_train['input_ids']).cuda(), \
    torch.tensor(tokens_train['attention_mask']).cuda(), \
    torch.tensor(train_labels).cuda())
val_data = TensorDataset(torch.tensor(tokens_val['input_ids']).cuda(), \
    torch.tensor(tokens_val['attention_mask']).cuda(), \
    torch.tensor(val_labels).cuda())
test_data = TensorDataset(torch.tensor(tokens_test['input_ids']).cuda(), \
    torch.tensor(tokens_test['attention_mask']).cuda(), \
    torch.tensor(test_labels).cuda())

train_dataloader = DistributedDataLoader(train_data, batch_size = batch_size, shuffle = True)
val_dataloader = DistributedDataLoader(val_data, batch_size = batch_size, shuffle = False)
test_dataloader = DistributedDataLoader(test_data, batch_size = batch_size, shuffle = False)
# splits = ['val', 'train', 'test']
# dataset = {}
# # 手动load test json
# for text, label in zip(train_texts, train_labels):
#     dataset['train'] = DATASET()
# test_path = f"{PATH_TO_DATASET}/BoolQ/test.jsonl"
# with open(test_path, encoding='utf8') as f:
#     lines = f.readlines()
#     for i, raw_row in enumerate(lines):
#         row = json.loads(raw_row)
#         label = 1 if row["label"]==True else 0
#         text_a = row['passage']
#         text_b = row['question']
#         template = (f'{text_a}. {text_b}',)

#         dataset['test'] = DATASET['BoolQ'](PATH_TO_DATASET, split, bmt.rank(), bmt.world_size(), tokenizer, max_encoder_length = PADDING_LEN).make_input(tokenizer, template, PADDING_LEN, label)

# train_dataloader = DistributedDataLoader(dataset['train'], batch_size = batch_size, shuffle=True)
# val_dataloader = DistributedDataLoader(dataset['val'], batch_size=batch_size, shuffle=False)

# optimizer and lr-scheduler
total_step = (len(train_dataloader)) * epochs
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
    for epoch in range(epochs):
        bmt.print_rank("Epoch {} begin...".format(epoch + 1))
        model.train()
        for step, data in enumerate(train_dataloader):
            input_ids, attention_mask, labels = data
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            global_loss = bmt.sum_loss(loss).item()
            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups,max_norm= float('inf'), scale = optimizer.scale, norm_type = 2)
            # bmt.optim_step(optimizer, lr_scheduler)
            bmt.optim_step(optimizer) # fixed learning rate
            if step % log_iter == 0:
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
                # input_ids = data["input_ids"]
                # attention_mask = data["attention_mask"]
                # labels = data["labels"]
                input_ids, attention_mask, labels = data
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
            early_stopping += 1
            bmt.print_rank(f"validation accuracy: {acc*100:.2f}\n")
            if acc > best_valid_acc:
                best_valid_acc = acc
                bmt.print_rank("saving the new best model...\n") # save checkpoint
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                early_stopping = 0 
            valid_acc_list.append(acc)
            if early_stopping > 5:
                bmt.print_rank("Accuracy have not rising for 5 epochs.Early stopping..")
                break # break for iter
    # 保存valid accuracy变化
    # df = pd.DataFrame(valid_acc_list)
    # df.to_csv(SAVED_PATH)
    bmt.print_rank(f"Validation accuracy is {acc}")

def check_performance():
    bmt.print_rank("Checking performance...\n")
    with torch.no_grad():
        pd = [] # prediction
        gt = [] # ground_truth
        for data in test_dataloader:
            input_ids, attention_mask, labels = data
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