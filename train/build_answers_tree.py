import os.path
import pandas as pd
import pickle
import tqdm


DATA_ROOT = '/home/egor/ml/DeepOverflow/data'
MAX_SCORE = 5


answers_tree = {}


posts_df = pd.read_csv(os.path.join(DATA_ROOT, 'incoming', 'Posts.csv'), encoding='ISO-8859-1', low_memory=False, dtype='str')

total = len(posts_df.index)


def parse_parent_id(id):
    return int(id.split('.')[0])


for idx, row in tqdm.tqdm(posts_df.iterrows(), total=total):
    if int(row['PostTypeId']) == 2:
        id = int(row['Id'])
        parent_id = parse_parent_id(row['ParentId'])
        score = int(row['Score'])
        if score > MAX_SCORE:
            if parent_id in answers_tree:
                answers_tree[parent_id].append(id)
            else:
                answers_tree[parent_id] = [id]

with open(os.path.join(DATA_ROOT, 'computed', 'answers_tree.pickle'), 'wb') as f:
    pickle.dump(answers_tree, f)