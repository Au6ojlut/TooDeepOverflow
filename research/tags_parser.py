import pandas as pd
import os
import pickle

posts_file = "Posts.csv"
path = "D:/data"

posts_info = pd.read_csv(os.path.join(path, posts_file), encoding="ISO-8859-1")

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
