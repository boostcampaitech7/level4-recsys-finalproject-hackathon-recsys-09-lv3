import os

import numpy as np
import random
import pandas as pd
import ast
from tqdm import tqdm
import pickle
import io
from nltk.tokenize import RegexpTokenizer

recomd = pd.read_csv('inter.csv')
recomd.rename(columns={"app_id": "item_id"}, inplace=True)

df_steam = recomd
df_steam['interaction'] = df_steam['is_recommended'].astype(int)

df_item = pd.read_csv('item.csv')
df_steam = df_steam.merge(df_item, on='item_id', how='left')

df_steam['hours'] = np.log1p(df_steam['hours'])

df_steam['date_release'] = pd.to_datetime(df_steam['date_release'])
df_steam['year'] = df_steam['date_release'].dt.year # 데이터 스플릿을 위한 Year 변수
df_steam['date_release'] = (df_steam['date_release'] - pd.Timestamp("1970-01-01")).dt.days
min_date = np.min(df_steam['date_release'])
max_date = np.max(df_steam['date_release'])

df_steam['date_release'] = df_steam['date_release'].map(lambda x: (x - min_date) / (max_date - min_date))

df_steam['positive_ratio'] = df_steam['positive_ratio'].map(lambda x: x / 100)

min_price_final = df_steam['price_final'].min()
max_price_final = df_steam['price_final'].max()
df_steam['price_final'] = (df_steam['price_final'] - min_price_final) / (max_price_final - min_price_final)

min_price_original = df_steam['price_original'].min()
max_price_original = df_steam['price_original'].max()
df_steam['price_original'] = (df_steam['price_original'] - min_price_original) / (max_price_original - min_price_original)

min_peak_ccu = df_steam['peak_ccu'].min()
max_peak_ccu = df_steam['peak_ccu'].max()
df_steam['peak_ccu'] = (df_steam['peak_ccu'] - min_peak_ccu) / (max_peak_ccu - min_peak_ccu)

df_steam["required_age"].value_counts().sort_index()
df_steam['required_age'] = df_steam['required_age'].map(lambda x: 1 if x >= 16 else 0)

min_price = df_steam['price'].min()
max_price = df_steam['price'].max()
df_steam['price'] = (df_steam['price'] - min_price) / (max_price - min_price)

df_steam['metacritic_score'] = df_steam['metacritic_score'].map(lambda x: x / 100)

df_steam['date'] = pd.to_datetime(df_steam['date'])
df_steam['date'] = (df_steam['date'] - pd.Timestamp("1970-01-01")).dt.days
min_date = np.min(df_steam['date'])
max_date = np.max(df_steam['date'])

df_steam['date'] = df_steam['date'].map(lambda x: (x - min_date) / (max_date - min_date))
df_steam = df_steam.sort_values(by='date')  # 시간순 정렬

user2count = df_steam.groupby(['item_id']).size().reset_index(name='count').sort_values(by='count')
item_ids = list(user2count['item_id'])
counts = np.array(user2count['count'])

df_steam = df_steam.join(user2count.set_index('item_id'), on='item_id')
min_count = np.min(df_steam['count'])
max_count = np.max(df_steam['count'])
df_steam['count'] = df_steam['count'].map(lambda x: (x - min_count) / (max_count - min_count))

# define unique function
def _unique(sample, fname):
    tmp_df = pd.DataFrame()
    tmp_df[fname] = sample[fname].unique()
    num = len(tmp_df)
    tmp_df['tmp_feature'] = range(num)
    sample = sample.join(tmp_df.set_index(fname), on=fname)
    sample.drop(fname, axis=1, inplace=True)
    sample = sample.rename(columns = {"tmp_feature": fname})
    return num, sample

num_user, df_steam = _unique(df_steam, 'user_id')
num_app, df_steam = _unique(df_steam, 'item_id')

# Reorder columns
# orders = ['user_id', 'item_id', 'hours', 'date_release', 'positive_ratio', 'price_final', 'price_original', 'peak_ccu', 'required_age', 'price', 'metacritic_score', 'date', 'count', 'combined_embedding', 'tag_embedding', 'category_embedding', 'interaction']
orders = ['user_id', 'item_id', 'hours', 'date_release', 'positive_ratio', 'price_final', 'price_original', 'peak_ccu', 'required_age', 'price', 'metacritic_score', 'date', 'count', 'interaction','year']
df_steam = df_steam[orders]
description = [
    ('user_id', num_user, 'spr'),  # 변환된 유저 ID
    ('item_id', num_app, 'spr'),  # 변환된 게임(아이템) ID
    ('hours', -1, 'ctn'),  # 플레이 시간
    ('date_release', -1, 'ctn'),  # 출시일
    ('positive_ratio', -1, 'ctn'),  # 긍정적인 평가 비율
    ('price_final', -1, 'ctn'),  # 최종 가격
    ('price_original', -1, 'ctn'),  # 원래 가격
    ('peak_ccu', -1, 'ctn'),  # 최대 동시 접속자 수
    ('required_age', 1 + np.max(df_steam["required_age"]), 'spr'),  # 필요한 나이
    ('price', -1, 'ctn'),  # 가격
    ('metacritic_score', -1, 'ctn'),  # 메타크리틱 점수
    ('date', -1, 'ctn'),  # 정규화된 날짜
    ('count', -1, 'ctn'),  # 리뷰 수
    # ('combined_embedding', len(df_steam["combined_embedding"].iloc[0]), 'emb'),  # title + description 결합 임베딩
    # ('tag_embedding', len(df_steam["tag_embedding"].iloc[0]), 'emb'),  # 태그 임베딩
    # ('category_embedding', len(df_steam["category_embedding"].iloc[0]), 'emb'),  # 카테고리 임베딩
    ('interaction', 2, 'label'),  # 클릭 여부
]

assert df_steam.isna().sum().sum() == 0, 'There are missing values in the preprocessed data!!'


def split_2(df_ratings, description, N=163, K=10):
    user2count = df_ratings.groupby(['item_id']).size().reset_index(name='count').sort_values(by='count')
    item_ids = user2count['item_id'].to_numpy()
    counts = user2count['count'].to_numpy()

    hot_item_ids = item_ids[counts > N]
    cold_item_ids = item_ids[(counts <= N) & (counts >= 3 * K)]

    item_group = df_ratings.groupby('item_id')

    train_base_list = []
    for item_id in hot_item_ids:
        df_hot = item_group.get_group(item_id).sort_values(by='date')
        train_base_list.append(df_hot)

    train_base = pd.concat(train_base_list, ignore_index=True)

    train_warm_a_list, train_warm_b_list, train_warm_c_list, test_list = [], [], [], []

    for item_id in cold_item_ids:
        df_cold = item_group.get_group(item_id).sort_values(by='date')
        df_cold = df_cold[df_cold['year'] >= 2020]
        train_warm_a_list.append(df_cold[:K])
        train_warm_b_list.append(df_cold[K:2 * K])
        train_warm_c_list.append(df_cold[2 * K:3 * K])
        test_list.append(df_cold[3 * K:])

    train_warm_a = pd.concat(train_warm_a_list, ignore_index=True)
    train_warm_b = pd.concat(train_warm_b_list, ignore_index=True)
    train_warm_c = pd.concat(train_warm_c_list, ignore_index=True)
    test = pd.concat(test_list, ignore_index=True)

    save_dic = {
        'train_base': train_base.sort_values('date'),
        'train_warm_a': train_warm_a.sort_values('date'),
        'train_warm_b': train_warm_b.sort_values('date'),
        'train_warm_c': train_warm_c.sort_values('date'),
        'test': test.sort_values('date'),
        'description': description
    }

    for name, df in save_dic.items():
        # with open('./emb_warm_split_{}.pkl'.format(name), 'bw+') as f:
        #     pickle.dump(df, f, protocol = pickle.HIGHEST_PROTOCOL)
        print("{} size: {}".format(name, len(df)))

    with open('./test_split.pkl', 'bw+') as f:
        pickle.dump(save_dic, f, protocol=pickle.HIGHEST_PROTOCOL)

split_2(df_steam, description)

# Get training data for Meta-Embedding method
with open('./test_split.pkl', 'rb') as f:
    data = pickle.load(f)

df_base = data['train_base']
item2group = df_base.groupby('item_id')

train_a_list, train_b_list, train_c_list, train_d_list = [], [], [], []

for item_id, df_group in tqdm(item2group):
    l, e = df_group.shape[0], df_group.shape[0] // 4
    train_a_list.append(df_group.iloc[0:e])
    train_b_list.append(df_group.iloc[e: 2 * e])
    train_c_list.append(df_group.iloc[2 * e: 3 * e])
    train_d_list.append(df_group.iloc[3 * e: 4 * e])

train_a = pd.concat(train_a_list, ignore_index=True)
train_b = pd.concat(train_b_list, ignore_index=True)
train_c = pd.concat(train_c_list, ignore_index=True)
train_d = pd.concat(train_d_list, ignore_index=True)

shuffle_idx = np.random.permutation(train_a.shape[0])
train_a = train_a.iloc[shuffle_idx]
train_b = train_b.iloc[shuffle_idx]
train_c = train_c.iloc[shuffle_idx]
train_d = train_d.iloc[shuffle_idx]

data["metaE_a"] = train_a
data["metaE_b"] = train_b
data["metaE_c"] = train_c
data["metaE_d"] = train_d

with open('./test_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("데이터 타입:", type(data))
print("데이터 크기:", len(data) if isinstance(data, dict) else "Not a dict")
print("데이터 키 목록:", data.keys() if isinstance(data, dict) else "Not a dict")
