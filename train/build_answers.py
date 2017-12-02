import os.path
import pandas as pd
import pickle
import tqdm


DATA_ROOT = '/home/egor/ml/DeepOverflow/data'

answers = {}


posts_df = pd.read_csv(os.path.join(DATA_ROOT, 'incoming', 'Posts.csv'), encoding='ISO-8859-1', low_memory=False, dtype='str')

total = len(posts_df.index)


for idx, row in tqdm.tqdm(posts_df.iterrows(), total=total):
    if int(row['PostTypeId']) == 2:
        id = int(row['Id'])
        text = row['Body']
        answers[id] = text

with open(os.path.join(DATA_ROOT, 'computed', 'answers.pickle'), 'wb') as f:
    pickle.dump(answers, f)