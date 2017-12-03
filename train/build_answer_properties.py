import os.path
import pandas as pd
import pickle
import tqdm


from deepoverflow.config import DATA_ROOT

answer_properties = {}


posts_df = pd.read_csv(os.path.join(DATA_ROOT, 'incoming', 'Posts.csv'), encoding='ISO-8859-1', low_memory=False, dtype='str')

total = len(posts_df.index)

entities_dict = {}

posts_df.loc[:, 'ViewCount'] = posts_df['ViewCount'].fillna(100)
posts_df.loc[:, 'Score'] = posts_df['Score'].fillna(0)

for idx, row in tqdm.tqdm(posts_df.iterrows(), total=total):
    if int(row['PostTypeId']) == 2:
        id = int(row['Id'])
        views = int(row['ViewCount'])
        score = int(row['Score'])

        answer_properties[id] = {'views': views, 'score': score}

with open(os.path.join(DATA_ROOT, 'computed', 'answer_properties.pickle'), 'wb') as handle:
    pickle.dump(answer_properties, handle, protocol=pickle.HIGHEST_PROTOCOL)