from cmath import inf
from msilib import sequence
import bmtrain as bmt
bmt.init_distributed(seed=0)
import torch
import numpy as np
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

from prepare_WNUT16 import WNUT16_dataset
ROBERTA_MODEL = 'roberta-base'
PADDING_LEN = 512

log_iter = 50
label_num= 21
epochs = 30
batch_size = 32
learning_rate = 1e-5
warm_up_ratio = 0.1
vocab_size = 50265

PATH_TO_DATASET = 'data'
BERT_PATH = '/root/bm_train_codes/save/roberta-base_original_wwm/checkpoints/checkpoint-18799.pt'
CHECKPOINT_PATH = 'saved_models/model.pt'
SAVED_PATH = 'saved_models/valid_acc.csv'

# bmt.print_rank("torch version", torch.__version__)
# bmt.print_rank(torch.cuda.get_arch_list())

class TokenClassificationModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = Roberta.from_pretrained(ROBERTA_MODEL) # load roberta from pretrained
        # bmt.load(self.roberta, BERT_PATH)
        print_inspect(self.roberta, "*")
        self.classifier = Linear(config.hidden_size, config.num_labels)
        bmt.init_parameters(self.classifier)

    def forward(self, input_ids, attention_mask = None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.classifier(outputs)
        return logits


config = RobertaConfig.from_pretrained(ROBERTA_MODEL)
model = RobertaModel(config)
# bmt.load(model, CHECKPOINT_PATH)

bmt.print_rank("Prepare dataset...")
bmt.print_rank(f"Local rank:{bmt.rank()}, World size:{bmt.world_size()}")
# train_texts, train_labels = get_WNUT16_data('WNUT16/wnut16_train.txt')
# val_texts, val_labels = get_WNUT16_data('WNUT16/wnut16_dev.txt')
# test_texts, test_labels = get_WNUT16_data('WNUT16/wnut16_test.txt')
train_dataset = WNUT16_Dataset('WNUT16/wnut16_train.txt')
val_dataset =  WNUT16_Dataset('WNUT16/wnut16_dev.txt')
test_dataset =  WNUT16_Dataset('WNUT16/wnut16_test.txt')



tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
# from tokenizers import Tokenizer
# from transformers import PreTrainedTokenizerFast
# tokenizer_obj = Tokenizer.from_file('tokenizer/tokenizer.json')
# tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
# tokenizer.pad_token = '[PAD]'
# tokenizer.eos_token = '[EOS]'
# tokenizer.sep_token = '[SEP]'
# tokenizer.mask_token = '[MASK]'
# tokenizer.sep_token = '[SEP]'

def batch_iter(dataset):
    st = 0
    input_ids_list = []
    labels_list = []
    while True:
        input_ids, attention_mask, labels = dataset[st]
        st += 1
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

        if len(input_ids_list) >= batch_size:
            yield {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "labels": torch.stack(labels_list)
            }
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

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
        pd = []
        gt = []
        for step, data in enumerate(batch_iter()):
            words,tags = data
            tags = np.array(tags.cuda())
            inpud_ids = words['input_ids'].cuda()
            attention_mask = words['attention_mask'].cuda()
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))
            clean_idx = (tags != -100)
            logits_clean = np.array(logits)[clean_idx]
            label_clean = tags[clean_idx]
            predictions = logits_clean.argmax(dim=1)
            
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