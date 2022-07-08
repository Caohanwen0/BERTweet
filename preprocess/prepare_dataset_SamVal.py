import os
from TweetNormalizer import *


def get_train_dataset():
    contents = []
    labels = []
    for root, dirs, files in os.walk("train"):
        for file in files:
            if file[0] == '.':
                continue
            with open("train/" + file, 'r') as f:
                raw = f.readlines()
                for line in raw:
                    if len(line) <= 1:
                        continue
                    split_line = line.split('\t') # 只切割一次
                    if len(split_line[-1]) <= 1:
                        split_line = split_line[:-1]
                    for i in range(len(split_line)):
                        if split_line[i].strip() in ['negative', 'positive', 'neutral']:
                            if split_line[i].strip() == "negative":
                                labels.append(0)
                            elif split_line[i].strip() == "neutral":
                                labels.append(1) 
                            else:
                                labels.append(2) 
                            contents.append(normalizeTweet(''.join(split_line[i+1:])))
                            break                 
    assert(len(contents)==len(labels))
    return contents, labels

def get_test_dataset():
    labels = []
    contents = []
    with open('test.txt', 'r')as f:
        raw = f.readlines()
        for line in raw:
            if len(line) <= 1:
                continue
            split_line = line.split('\t') # 只切割一次
            if len(split_line[-1]) <= 1:
                split_line = split_line[:-1]
            for i in range(len(split_line)):
                if split_line[i].strip() in ['negative', 'positive', 'neutral']:
                    if split_line[i].strip() == "negative":
                        labels.append(0)
                    elif split_line[i].strip() == "neutral":
                        labels.append(1) 
                    else:
                        labels.append(2) 
                    contents.append(normalizeTweet(''.join(split_line[i+1:])))
                    break 
    assert(len(contents)==len(labels))
    return contents, labels 
