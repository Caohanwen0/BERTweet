import json,os,random
def get_Article_Bias_dataset(json_file_dir):
    contents,labels=[], []
    filename_list = [os.path.join(json_file_dir, filename) for filename in os.listdir(json_file_dir)]
    for filename in filename_list:
        with open(filename, 'r')as f:
            js = json.load(f)
            contents.append(js['content'])
            labels.append(int(js['bias']))
    assert len(contents)==len(labels)
    test_contents = contents[:len(contents)//10]
    test_labels = labels[:len(contents)//10]
    temp_contents = contents[len(contents)//10:] 
    temp_labels = labels[len(contents)//10:]  
    return temp_contents, temp_labels, test_contents, test_labels
