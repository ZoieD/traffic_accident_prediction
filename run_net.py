import pandas as pd
import pickle
import numpy as np

import h5py
import pydot
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import RMSprop, Adagrad, Adam
from keras.utils import plot_model
from keras.losses import binary_crossentropy
#from keras.metrics import precision as prec_metric
import keras.backend as K

import xgboost
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold,StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score, roc_curve, average_precision_score,precision_recall_curve
# For plotting
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",80)

df = pd.read_csv('./data/final_data.csv')
df = df.dropna(how='any',axis=0)
print(df)

one_hot_field = ['hour', 'DAY_OF_WEEK', 'month', 'Light_Condition', 'ROAD_TYPE', 'wind_dir', 'SURFACE_COND', 'NODE_TYPE', 'Deg_Urban_Name']

# One-Hot encode a couple of variables
df_one_hot = pd.get_dummies(df,columns=one_hot_field)

# Get the one-hot variable names
one_hot_feature_names = pd.get_dummies(df[one_hot_field],columns=one_hot_field).columns.tolist()
df_one_hot.head()

y = df['target'].values
float_feature_names = [
    'SPEED_ZONE',
    'wind_speed',
    'temperature',
]
float_features = df_one_hot.xs(float_feature_names,axis=1).values
# Use scikit-learn's StandardScaler
scaler = StandardScaler()
float_scaled = scaler.fit_transform(float_features)
#print (float_features.mean(axis=0))

df_one_hot[float_feature_names] = float_scaled

with open('scalers.pkl','wb') as fp:
    pickle.dump(scaler,fp)

binary_feature_names = [
    'snowing',
    'raining',
    'foggy',
    'smoke',
    'dust',
    'strong_winds',
]
df_one_hot = df_one_hot.xs(float_feature_names + binary_feature_names + one_hot_feature_names,axis=1)
df_one_hot.head()

X = df_one_hot.values
y = df['target'].values
feature_names = df_one_hot.columns.tolist()

wrangler = {
    'scaler': scaler,
    'float_feature_names': float_feature_names,
    'one_hot_field': one_hot_field,
    'feature_names': feature_names
}
with open('wrangler_new.pkl','wb') as fp:
    pickle.dump(wrangler,fp)

# create model
model = Sequential()
model.add(Dense(25, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

main_input = Input((X.shape[1],),name='input')

dense_00 = Dense(X.shape[1], activation='relu')(main_input)
add_00 = Add(name='add_input_1')([dense_00, main_input])
dropout_00 = Dropout(0.1)(add_00)
dense_01 = Dense(X.shape[1], activation='relu')(dropout_00)
add_01 = Add(name='add_input_2')([dense_01, main_input])
dropout_01 = Dropout(0.1)(add_01)
dense_02 = Dense(X.shape[1], activation='relu')(dropout_01)
add_02 = Add(name='add_input_3')([dense_02, main_input])
dropout_02 = Dropout(0.1)(add_02)
dense_03 = Dense(X.shape[1], activation='relu')(dropout_02)
add_03 = Add(name='add_input_4')([dense_03, main_input])
dropout_03 = Dropout(0.1)(add_03)
dense_04 = Dense(X.shape[1], activation='relu')(dropout_03)
add_04 = Add()([dense_04, main_input])
dropout_04 = Dropout(0.1)(add_04)
dense_05 = Dense(50,activation='relu')(dropout_04)
dense_06 = Dense(25,activation='relu')(dense_05)
dense_07 = Dense(15,activation='relu')(dense_06)
output = Dense(1,activation='sigmoid')(dense_07)

model = Model(inputs=[main_input],outputs=[output])

optimizer = Adam()
model.compile(loss='binary_crossentropy', optimizer=optimizer)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from time import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')

sample_weight = ((4.0*y_train > 0) + 1.0*(y_train == 0))/5.0

model.fit(X_train, y_train,
          validation_data=(X_test,y_test),
          epochs=100,
          verbose=1,
          shuffle=True,
          batch_size=100000,
          sample_weight = sample_weight,
          callbacks=[tensorboard,early_stopping])

model.save('logloss.h5')

y_pred = model.predict(X_test)

ypt = y_pred>0.1
'''
print ('Test Accuracy:',accuracy_score(y,ypt))
print ('Test F1:',f1_score(y,ypt))
print ('Test Precision:',precision_score(y,ypt))
print ('Test AP:',average_precision_score(y,ypt))
print ('Test Recall:',recall_score(y,ypt))
'''
#'''
print ('Test Accuracy:',accuracy_score(y_test,ypt))
print ('Test F1:',f1_score(y_test,ypt))
print ('Test Precision:',precision_score(y_test,ypt))
print ('Test AP:',average_precision_score(y_test,ypt))
print ('Test Recall:',recall_score(y_test,ypt))
#'''

precision,recall,thresholds = precision_recall_curve(y_test,y_pred)
#precision,recall,thresholds = precision_recall_curve(y,y_pred)

plt.plot(thresholds,precision[:-1],'r-',label='P (DNN)',color='orange',lw=3)
plt.plot(thresholds,recall[:-1],'r-',label='R (DNN)',color='steelblue',lw=3)
plt.gca().set_xbound(lower=0,upper=1)
plt.grid()
plt.legend()

plt.figure(figsize=(15,15))

precision,recall,thresholds = precision_recall_curve(y_test,y_pred)
#precision,recall,thresholds = precision_recall_curve(y,y_pred)

fig,ax = plt.subplots()
plt.plot(precision,recall,label='Precision vs Recall',lw=3)

ax.set_xbound(lower=0.0,upper=1.0)
ax.set_ybound(lower=0.01,upper=1.0)
plt.grid()
plt.legend()

plt.figure()
plt.hist(y_pred,50)