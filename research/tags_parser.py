import pandas as pd
import os
import pickle
from deepoverflow.config import DATA_ROOT

posts_path = os.path.join(DATA_ROOT, 'incoming', 'Posts.csv')
posts_info = pd.read_csv(posts_path, encoding="ISO-8859-1", low_memory=False)

posts = posts_info[['Id', 'Tags', 'PostTypeId']]

id_to_tags = {}
for index, row in posts.iterrows():
    if int(row['PostTypeId']) == 2:
        continue
    str_idx = int(row['Id'])
    if row['Tags'] is not None and row['Tags'] != '':
        id_to_tags[str_idx] = row['Tags']

with open('id_to_tags.pickle', 'wb') as handle:
    pickle.dump(id_to_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
