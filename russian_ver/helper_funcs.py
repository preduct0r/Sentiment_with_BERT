import json
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import time
import datetime


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def get_rutweetcorp_data():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv(r'C:\Users\denis\PycharmProjects\Sentiment_with_BERT\data\RuTweetCorp\positive.csv',
                                sep=';', error_bad_lines=False, names=n, usecols=['text'])
    data_negative = pd.read_csv(r'C:\Users\denis\PycharmProjects\Sentiment_with_BERT\data\RuTweetCorp\negative.csv',
                                sep=';', error_bad_lines=False, names=n, usecols=['text'])

    sample_size = min(data_positive.shape[0], data_negative.shape[0])
    raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                               data_negative['text'].values[:sample_size]), axis=0)
    labels = [1] * sample_size + [0] * sample_size

    data = [preprocess_text(t) for t in raw_data]

    df_train = pd.DataFrame(columns=['Text', 'Label'])
    df_test = pd.DataFrame(columns=['Text', 'Label'])

    df_train['Text'], df_test['Text'], df_train['Label'], df_test['Label'] = train_test_split(data, labels,
                                                                                              test_size=0.2,
                                                                                              random_state=1)

    return df_train, df_test



def get_data_kaggle_news():

    train_file = r'C:\Users\denis\PycharmProjects\Sentiment_with_BERT\data\sentiment-analysis-in-russian\train.json'
    x_train, y_train = [], [], [], []


    with open(train_file, encoding='utf-8') as json_file:
        data = json.load(json_file)

        for row in data:
            sentiment = -1

            if row['sentiment'] == 'negative':
                sentiment = 0
            elif row['sentiment'] == 'neutral':
                sentiment = 1
            else:
                sentiment = 2

            if sentiment == -1:
                continue

            x_train.append(row['text'])
            y_train.append(sentiment)

    print('Train sentences: {}'.format(len(x_train)))
    print('Train labels: {}'.format(len(y_train)))

    return x_train, y_train


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))