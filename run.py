import pandas as pd
import pickle
import numpy as np
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
# df = pd.read_csv('../data/positive_feature.csv')
df = df.dropna(how='any',axis=0)

one_hot_field = ['hour', 'DAY_OF_WEEK', 'month', 'Light_Condition', 'ROAD_TYPE', 'wind_dir', 'SURFACE_COND', 'NODE_TYPE', 'Deg_Urban_Name']
# one_hot_field = ['DAY_OF_WEEK']
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
# df_one_hot = df_one_hot.xs(float_feature_names + one_hot_feature_names,axis=1)

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

feature_sel = range(len(feature_names))
#feature_sel = [-1,-2,-3]
Xs = X[:,feature_sel]
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.1)#, random_state=2)
fnames = np.array(feature_names)[feature_sel]

dtrain = xgboost.DMatrix(X_train,label=y_train,feature_names=fnames)
dtest =  xgboost.DMatrix(X_test,label=y_test,feature_names=fnames)

params = {
    'max_depth':6,
    'min_child_weight': 5.0,
    'reg_lambda': 1.0,
    'reg_alpha':0.0,
    'scale_pos_weight':1.0,
    'eval_metric':'auc',
    'objective':'binary:logistic',
    'eta':0.5
}

booster = xgboost.train(params,dtrain,
    evals = [(dtest, 'eval')],
    num_boost_round=3000,
    early_stopping_rounds=25
)

print(fnames)

plt.figure(figsize=(15,15))
xgboost.plot_importance(booster,ax=plt.gca(),importance_type='weight')

booster.save_model('new_0001.model')

y_pred_test = booster.predict(dtest)

fpr, tpr, thresholds = roc_curve(y_test,y_pred_test)

y_pred_train = booster.predict(dtrain)
fpr_train, tpr_train, thresholds_train = roc_curve(y_train,y_pred_train)
fig,ax = plt.subplots()
plt.plot([0,1],[0,1],'r-',label='Random Guess',color='orange',lw=3)
plt.plot(fpr,tpr,label='ROC (Test)',lw=3)
plt.plot(fpr_train,tpr_train,'r:',label='ROC (Train)',color='steelblue',lw=3)
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('./outputs/roc.png',bbox_inches='tight')

plt.plot(thresholds,tpr,'r-',label='TPR (Test)',color='orange',lw=3)
plt.plot(thresholds_train,tpr_train,'r:',label='TPR (Train',color='orange',lw=3)
plt.plot(thresholds,fpr,'r-',label='FPR (Test)',color='steelblue',lw=3)
plt.plot(thresholds_train,fpr_train,'r:',label='FPR (Train)',color='steelblue',lw=3)
plt.gca().set_xbound(lower=0,upper=1)
plt.xlabel('Threshold')
plt.ylabel('True/False Positive Rate')
plt.legend()
plt.savefig('./outputs/tpr_fpr.png',bbox_inches='tight')

plt.figure(figsize=(15,15))

y_pred_test = booster.predict(dtest)
y_pred_train = booster.predict(dtrain)

precision,recall,thresholds = precision_recall_curve(y_test,y_pred_test)
precision_train, recall_train, thresholds_train = precision_recall_curve(y_train,y_pred_train)
fig,ax = plt.subplots()
plt.plot(precision,recall,label='PR (Test)',lw=3)
plt.plot(precision_train,recall_train,label='PR (Train)',lw=3)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.grid()
plt.legend()
plt.savefig('./outputs/pr_curve.png',bbox_inches='tight')
plt.matplotlib.__version__

plt.plot(thresholds,precision[:-1],'r-',label='P (Test)',color='orange',lw=3)
plt.plot(thresholds_train,precision_train[:-1],'r:',label='P (Train',color='orange',lw=3)
plt.plot(thresholds,recall[:-1],'r-',label='R (Test)',color='steelblue',lw=3)
plt.plot(thresholds_train,recall_train[:-1],'r:',label='R (Train)',color='steelblue',lw=3)
#plt.plot([0,1],[0,1],'k-',lw=2)
plt.gca().set_xbound(lower=0,upper=1)
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()

