import pandas as pd

def get_Dreaddit_dataset(csv_path):
    data = pd.read_csv(csv_path,usecols=['text','label'],dtype = {'text':str, 'label':int})
    return list(data['text']), list(data['label'])