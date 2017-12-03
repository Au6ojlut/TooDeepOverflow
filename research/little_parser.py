import pandas as pd
import os
import pickle
import math

posts_file = "Posts.csv"
path = "D:/data"

posts_info = pd.read_csv(os.path.join(path, posts_file), encoding="ISO-8859-1")

posts = posts_info[['Id', 'AcceptedAnswerId', 'PostTypeId']]

accepted_answer_ids = {}
for index, row in posts.iterrows():
    if int(row['PostTypeId']) == 2:
        continue
    str_idx = int(row['Id'])
    if row['AcceptedAnswerId'] is not None and not math.isnan(row['AcceptedAnswerId']) and row['AcceptedAnswerId'] != '':
        accepted_answer_ids[str_idx] = int(row['AcceptedAnswerId'])

with open('id_to_acceptedAnswer.pickle', 'wb') as handle:
    pickle.dump(accepted_answer_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
