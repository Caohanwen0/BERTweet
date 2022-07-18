import json
def get_incivility_dataset(json_path):
    contents, labels = [], []
    with open(json_path) as f:
        raw = f.readlines()
    for line in raw:
        js = json.loads(line)
        contents.append(js['comment'])
        labels.append(int(js['label']))
    assert len(contents) == len(labels)
    return contents, labels