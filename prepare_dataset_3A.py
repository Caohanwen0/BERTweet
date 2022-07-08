import os
from TweetNormalizer import *

def get_3A_dataset(file_path, do_preprocess):
    labels = []
    contents = []
    with open(file_path, 'r')as f:
        raw = f.readlines()
        for line in raw:
            if len(line) <= 1:
                continue
            split_line = line.split('\t') # 只切割一次
            if len(split_line[-1]) <= 1:
                split_line = split_line[:-1]
            for i in range(1, len(split_line)):
                try:
                    if int(split_line[i].strip()) in [0,1]:
                        if int(split_line[i].strip()) == 0:
                            labels.append(0)
                        else:
                            labels.append(1)
                        if not do_preprocess:
                            contents.append(normalizeTweet(''.join(split_line[i+1:])))
                        else:
                            contents.append(preprocess_clean(''.join(split_line[i+1:]))) 
                        break
                except:
                    continue
    assert(len(contents)==len(labels))
    return contents, labels 
