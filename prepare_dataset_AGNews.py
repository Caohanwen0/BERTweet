import pandas as pd

def get_AGNews_dataset(csv_path):
    data = pd.read_csv(csv_path,dtype = {'Class Index':int, 'Title':str, 'Description':str})
    contents = [title + description for title, description in zip(list(data['Title']),list(data['Description']))]
    return contents, list(data['Class Index'])