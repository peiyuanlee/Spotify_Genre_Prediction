import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing

def preprocessing_data(df):
    df.dropna(axis=0,inplace=True)
    df.drop('track_id', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop_duplicates(subset = ['popularity', 'track_name', 'album_name', 'tempo'], keep='first', inplace= True)

    label_encoder = preprocessing.LabelEncoder()
    df['explicit']=label_encoder.fit_transform(df['explicit'])

    df = df.drop(columns=['album_name','track_name','artists', 'Unnamed: 0'])
    top20 = df['track_genre'].value_counts(ascending=False)[:20].index
    df= df[df['track_genre'].isin(top20)]

    ss = preprocessing.StandardScaler()
    num_cols = df.drop(columns='track_genre').columns
    df[num_cols] = ss.fit_transform(df.drop(columns='track_genre'))

    return df