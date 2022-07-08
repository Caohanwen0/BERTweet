
cls_token = "<s>"
sep_token = "</s>"
pad_token = "<pad>"
tag_to_id = {
        'B-company':0,
        'I-company':1,
        'B-geo-loc':2,
        'I-geo-loc':3,
        'B-product':4,
        'I-product':5,
        'B-musicartist':6,
        'I-musicartist':7,
        'B-sportsteam':8,
        'I-sportsteam':9,
        'B-person':10,
        'I-person':11,
        'B-facility':12,
        'I-facility':13,
        'B-movie':14,
        'I-movie':15,
        'B-tvshow':16,
        'I-tvshow':17,
        'O':18,
        'B-other':19,
        'I-other':20,
    }

id_to_tag={v:k for k,v in enumerate(tag_to_id)}

def align_label_example(tokenized_input, labels, label_all_tokens):

        word_ids = tokenized_input.word_ids()

        previous_word_idx = None
        label_ids = []
   
        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)
                
            elif word_idx != previous_word_idx:
                try:
                  label_ids.append(tag_to_id[labels[word_idx]])
                except:
                  label_ids.append(-100)
        
            else:
                label_ids.append(tag_to_id[labels[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
      

        return label_ids


class WNUT16_Dataset:
    def __init__(self, file_path, tokenizer):
        s,t = get_WNUT16_data(file_path)
        self.sents = s
        self.tags = t
        self.tag_to_id = tag_to_id
        self.tokenizer = tokenizer
        self.text_tokenized = []
        for sent in self.sents:
            self.text_tokenized.append(tokenizer(sent, padding='max_length', \
                max_length=512, truncation=True, return_tensors="pt"))
        self.new_label = []
        for txt, label in zip(self.text_tokenized, self.tags):
            self.new_label.append(align_label_example(txt, label, label_all_tokens=True))

    def __len__(self):
        return len(self.sents)
            
    def __getitem__(self, idx):
        words, tags = self.text_tokenized[idx], self.new_label[idx] # words, tags: string list
        return words,tags

def get_WNUT16_data(file_path):
    sents, tags = [], []
    with open(file_path, 'r')as f:
        raw = f.readlines()
        current_sent = []
        current_tag = []
        for line in raw:
            if len(line) <= 1: # 一个句子结束了
                if len(current_sent)>0:
                    assert len(current_sent) == len(current_tag)
                    sents.append([cls_token]+current_sent+[sep_token])
                    tags.append([pad_token]+current_tag+[pad_token])
                    current_sent = []
                    current_tag = []
            else:
                line = line.split('\t')
                assert len(line)==2
                current_sent.append(line[0]) #word
                current_tag.append(tag_to_id[line[1].strip()]) # tags
    if len(current_sent)>0:
        assert len(current_sent) == len(current_tag)
        sents.append([cls_token]+current_sent+[sep_token])
        tags.append([pad_token]+current_tag+[pad_token]) 
    assert len(tags)==len(sents)
    return sents, tags
                

                 

