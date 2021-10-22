import pandas as pd
import yaml
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy
from keras.callbacks import Callback

from keras import backend as K
from keras.backend import clear_session

import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('nn_result.png')

clear_session()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


start = datetime.now()
print('Running the Neural Net Code ... ')
print('')

with open('../config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

SELECTED_DATASET = config['DATASET']
SELECTED_MODEL = config['MODEL']
train_features = pd.read_csv('../' + SELECTED_DATASET + '_train_features.csv')
test_features = pd.read_csv('../' + SELECTED_DATASET + '_test_features.csv')


train_types = []

for row in train_features['Type']:
    if row == 'Class':
        train_types.append(1)
    else:
        train_types.append(0)
        
train_features['Type_encode'] = train_types

test_types = []

for row in test_features['Type']:
    if row == 'Class':
        test_types.append(1)
    else:
        test_types.append(0)
        
test_features['Type_encode'] = test_types

X_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
y_train = train_features['Match']

X_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
y_test = test_features['Match']

df_train = train_features.loc[:, 'Ngram1_Entity':'Type_encode']
df_train['Match'] = train_features['Match']

df_test = test_features.loc[:, 'Ngram1_Entity':'Type_encode']
df_test['Match'] = test_features['Match']

X_train = X_train.fillna(value=0)
X_test = X_test.fillna(value=0)



model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=3, padding="same"))
model.add(Dense(32, input_dim=88, activation='relu'))
# model.add(LSTM(4))
# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  

# compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

# fit the model
# model.fit(X_train, y_train, epochs=15, validation_data = (X_test, y_test), callbacks=[checkpoint], verbose=True)
history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, callbacks=[checkpoint], verbose=True)
plot_history(history)
# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)

print('================================================')
print('loss: {:.4f}'.format(loss))
print('accuracy: {:.4f}'.format(accuracy))
print('f1_score: {:.4f}'.format(f1_score))
print('precision: {:.4f}'.format(precision))
print('recall: {:.4f}'.format(recall))
print('================================================')
end = datetime.now()
print('Total time elapsed: ', (end - start).total_seconds() )