import sys
import numpy as np
import scipy.stats as stats
import pylab as plt
import webcolors
import pickle
from keras.models import load_model
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import preprocessing
from keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape

maxlen = 6
with open('word2rgb_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

index_max = len(tokenizer.word_index)+1

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, index_max)))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.load_weights('word2rgb_lstm_weights.h5')

# plot a color image
def plot_rgb(rgb):
    data = [[rgb]]
    plt.figure(figsize=(2,2))
    plt.imshow(data, interpolation='nearest')
    plt.show()

def scale(n):
    return int(n * 255) 

def predict(name):
    name = name.lower()
    tokenized = tokenizer.texts_to_sequences([name])
    padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=maxlen)
    one_hot = np_utils.to_categorical(padded, num_classes=index_max)
    pred = model.predict(np.array(one_hot))[0]
    r, g, b = scale(pred[0]), scale(pred[1]), scale(pred[2])
    print(name + ',', 'R,G,B:', r,g,b)
    plot_rgb(pred)

predict("红")
predict("浅红")
predict("浅浅红")