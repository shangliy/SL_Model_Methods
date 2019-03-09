import sys
import numpy as np
import scipy.stats as stats
import pylab as plt
import webcolors
import pickle
from bs4 import BeautifulSoup

color_dict = {}

soup = BeautifulSoup(open("中国色 － 中国传统颜色.html"), "html.parser")

color_ul = soup.find('ul', id='colors')
color_ul_lis = color_ul.find_all('li')
color_count = 0
for li in color_ul_lis:
    color_name = li.find('span',class_='name').text
    #print(color_name)
    
    color_hex = li.find('span',class_='rgb').text
    
    rgb = webcolors.hex_to_rgb('#'+color_hex)
    red = rgb.red
    green = rgb.green
    blue = rgb.blue
    #print(red, green, blue)
    color_dict[color_name] = [red, green, blue]
    color_count += 1
print(color_count)

soup = BeautifulSoup(open("中文颜色名称颜色对照表 - 明子健 - ITeye博客.html"), "html.parser")

color_ul_lis = soup.find_all('td')

for li in color_ul_lis:
    color_name, color_hex = li.text.split()
    #print(color_name)
    rgb = webcolors.hex_to_rgb(color_hex)
    red = rgb.red
    green = rgb.green
    blue = rgb.blue
    #print(red, green, blue)
    color_dict[color_name] = [red, green, blue]
    color_count += 1

print(color_count)


soup = BeautifulSoup(open("命名顏色代碼.html"), "html.parser")

color_ol = soup.find('ol')
color_ul_lis = color_ol.find_all('li')

for li in color_ul_lis:
    color_name, color_hex = (li.find('a').text.split('#'))
    #print(color_name.split('／'))
    color_name_list = color_name.split('／')
    rgb = webcolors.hex_to_rgb('#'+color_hex)
    red = rgb.red
    green = rgb.green
    blue = rgb.blue
    #print(red, green, blue)
    for c in color_name_list:
        color_dict[c] = [red, green, blue]
    color_count += 1

with open('color_list.txt') as f:
    data = f.readlines()
row_num = len(data) + 1
for i in range(int(row_num/3)):
    color_name = data[3*i].strip()
    #print(color_name)
    color_hex = data[3*i+1].strip()
    rgb = webcolors.hex_to_rgb(color_hex)
    red = rgb.red
    green = rgb.green
    blue = rgb.blue
    #print(red, green, blue)
    color_dict[color_name] = [red, green, blue]
    color_count += 1

print(color_count)
print(len(color_dict))

key_len_list = []
for key in color_dict:
    key_len_list.append(len(key))
    #sys.exit()
print(max(key_len_list))

names = list(color_dict.keys())
data_ = {}
data_["red"] = []
data_["green"] = []
data_["blue"] = []
for i in range(len(names)):
    data_["red"].append(color_dict[names[i]][0])
    data_["green"].append(color_dict[names[i]][1])
    data_["blue"].append(color_dict[names[i]][2])
    

data = {}
data["red"] = np.array(data_["red"])
data["green"] = np.array(data_["green"])
data["blue"] = np.array(data_["blue"])

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import preprocessing

maxlen = 6
t = Tokenizer(char_level=True)
t.fit_on_texts(names)
index_max = len(t.word_index)+1

tokenized = t.texts_to_sequences(names)
padded_names = preprocessing.sequence.pad_sequences(tokenized, maxlen=maxlen)

from keras.utils import np_utils
one_hot_names = np_utils.to_categorical(padded_names)


def norm(value):
    return value / 255.0

normalized_values = np.column_stack([norm(data["red"]), norm(data["green"]), norm(data["blue"])])

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Reshape

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, index_max)))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

history = model.fit(one_hot_names, normalized_values,
                    epochs=50,
                    batch_size=32)
#model.save('word2rgb_lstm.h5')
model.save('word2rgb_lstm_weights.h5')

with open('word2rgb_tokenizer.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    tokenized = t.texts_to_sequences([name])
    padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=maxlen)
    one_hot = np_utils.to_categorical(padded, num_classes=index_max)
    pred = model.predict(np.array(one_hot))[0]
    r, g, b = scale(pred[0]), scale(pred[1]), scale(pred[2])
    print(name + ',', 'R,G,B:', r,g,b)
    plot_rgb(pred)

predict("卡其色")
#predict("forest")
#predict("keras red")
