import numpy as np
import re
import itertools
from collections import Counter
import csv
import sys
csv.field_size_limit(sys.maxsize)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def getAllAuthors():

    authors = []
    with open('./data/rt-polaritydata/winereviews.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')

        for row in reader:
            if len(row[20].split(" ")) <= 200:
                authors.append(row[19])
    result = list(set(authors))
    return result


def load_data_and_labels(author):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    x_text = []
    scores = []

    with open('./data/rt-polaritydata/winereviews.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')

        for row in reader:
            if len(row[20].split(" ")) <= 200 and author in row[19]:
                x_text.append(row[20])
                scores.append(float(row[18]))



    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    y = scores
    return [x_text, y]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
