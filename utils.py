import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import re
from bs4 import BeautifulSoup
import lxml

from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, LSTM, Bidirectional
from tensorflow.keras.models import Model
from nltk.corpus import stopwords
import tokenization




model_nn = load_model("model_nn")
bert_model = load_model("model_bert")
bert_layer = hub.KerasLayer(hub.load("bert_layer_downloaded"))
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
stop_words = set(stopwords.words('english'))
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
def preprocess_text(text):
    #text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    filtered_text = []
    for i in text.split():
        i = i.lower()
        if i not in stop_words:
            filtered_text.append(i)
    text = re.sub(r"http\S+", "", text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub("\S*\d\S*", "", text).strip()
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = re.sub('<.*?>','',text)
    return text
def tokenization(X):
    max_seq_len = 50
    token_train = tokenizer.tokenize(X)
    token_train = token_train[0: (max_seq_len-2)]
    token_train = ['[CLS]', *token_train, '[SEP]']
    len_before = len(token_train)
    if (len(token_train) < max_seq_len):
        for i in range (len(token_train), 50):
            token_train.append('[PAD]')
    token_train = tokenizer.convert_tokens_to_ids(token_train)
    mask_train = ([1]*len_before + [0]* (max_seq_len - len_before))
    segment_train = [0]*max_seq_len
    
    return token_train, mask_train, segment_train



def sentiment_analysis(review_text):
    review_text = preprocess_text(review_text)
    X_tokens = []
    X_mask = []
    X_segment =[]
    tokens, mask, segment = tokenization(review_text)
    X_tokens.append(tokens)
    X_mask.append(mask)
    X_segment.append(segment)
    X_tokens =np.array(X_tokens)
    X_mask = np.array(X_mask)
    X_segment = np.array(X_mask)
    X_pooled_output = bert_model.predict([X_tokens, X_mask, X_segment])
    prediction = model_nn.predict(X_pooled_output)
    prediction_list = list(prediction[0])
    max_value = max(prediction_list)
    max_index = prediction_list.index(max_value)
    sentiment = "Null"
    if (max_index == 0):
        sentiment = "Negative"
    elif(max_index == 1):
        sentiment = "Positive"
    else:
        sentiment = "Neutral"



    




    return sentiment