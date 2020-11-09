# Importing all the required libraries.
#-------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import lightgbm as lgb

"""Harini"s Contribution"""

# Pre-processing and initialising all the data.
#-------------------------------------------------------------------------------------------------------------------

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 50000


def handle_missing_inplace(dataset): 
    dataset['category_name'].fillna(value='missing', inplace=True) 
    dataset['brand_name'].fillna(value='missing', inplace=True) 
    dataset['item_description'].replace('No description yet,''missing', inplace=True) 
    dataset['item_description'].fillna(value='missing', inplace=True)
def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]  #Finding the index of the rows where brand name is not missing
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing' #Replace the value as 'missing' for the rows which are not a part of pop_brand
def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['brand_name'] = dataset['brand_name'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')
    
df = pd.read_csv('Dataset.csv')
msk = np.random.rand(len(df)) < 0.8 
train = df[msk]
test = df[~msk]
test_new = test.drop('price', axis=1)
y_test = np.log1p(test["price"])
train = train[train.price != 0].reset_index(drop=True)

nrow_train = train.shape[0]
y = np.log1p(train["price"])
merge: pd.DataFrame = pd.concat([train, test_new], sort=False)  #Concatenating the training and test data

handle_missing_inplace(merge)
cutting(merge)
to_categorical(merge)

# Encoding all the string data to proper ints.
#-------------------------------------------------------------------------------------------------------------------

cv1 = CountVectorizer(min_df=NAME_MIN_DF)
X_name = cv1.fit_transform(merge['name'])
X_name
cv2 = CountVectorizer()
X_category = cv2.fit_transform(merge['category_name'])

tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION, ngram_range=(1, 3), stop_words='english')
X_description = tv.fit_transform(merge['item_description'])

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])

X_dummies=pd.get_dummies(merge[['item_condition_id', 'shipping']], sparse=True).values
X_dummies = csr_matrix(X_dummies.astype(int))
sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

# Fitting the model and predicting the final values.
#-------------------------------------------------------------------------------------------------------------------

train_X =lgb.Dataset(X, label=y)
params = {
        'learning_rate': 0.75, #impact of each tree in final prediction
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }

gbm = lgb.train(params, train_set=train_X, num_boost_round=3200, verbose_eval=100)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
f1=np.expm1(y_test)
final=np.expm1(y_pred)

"****************************************************************"

# Flask code to get data from form and then predict values according to the passed data with proper type casting.
#------------------------------------------------------------------------------------------------------------------

""" Yash's Contribution """
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/',methods = ['POST','GET'])
def getValue():
   if (request.method == 'POST'): 
       
       """Akshita's Contribtuion"""
       
       train_id=request.form.get('train_id')
       #print(train_id)
       name=request.form.get('name')
       #print(name+'Tester')
       item_condition_id=request.form.get('item_condition_id')
       #print(item_condition_id)
       category=request.form.get('category')
       #print(category+'Tester')
       brand_name=request.form.get('brand')
       #print(brand_name+'Tester')
       shipping=request.form.get('shipping')
       #print(shipping)
       item_description=request.form.get('item_description')
       #print(item_description+'Tester')
     
             
       train_new = df[msk]       
       test_1 = df[~msk]
       test_new_new = test_1.drop('price', axis=1)
       #Problem was because of ints getting passed as strings. So, type casted it.
       test_new_new=test_new_new.append({'train_id':int(train_id), 'name': name, 'item_condition_id':int(item_condition_id), 'category_name': category,'brand_name': brand_name, 'shipping': int(shipping), 'item_description': item_description}, ignore_index=True)
       train_new = train_new[train_new.price != 0].reset_index(drop=True)
       tnrow_train = train_new.shape[0]
       tmerge: pd.DataFrame = pd.concat([train_new, test_new_new], sort=False)
       
       
       handle_missing_inplace(tmerge)
       cutting(tmerge)
       to_categorical(tmerge)
       
        
       t_name = cv1.transform(tmerge['name'])        
       t_category = cv2.transform(tmerge['category_name'])        
       t_description = tv.transform(tmerge['item_description'])        
       t_brand = lb.transform(tmerge['brand_name'])        
       t_dummies=pd.get_dummies(tmerge[['item_condition_id', 'shipping']], sparse=True).values        
       t_dummies = csr_matrix(t_dummies.astype(int))        
       k = hstack((t_dummies, t_description, t_brand, t_category, t_name)).tocsr()        
       tmask = np.array(np.clip(k.getnnz(axis=0) - 1, 0, 1), dtype=bool)        
       k = k[:, tmask]       
       t = k[tnrow_train:] 
        
       ty_pred = gbm.predict(t, num_iteration=gbm.best_iteration)        
       tfinal=np.expm1(ty_pred)
       
       """*******************************************************"""
       
       return render_template('Answer.html', ans = tfinal[-1])      
       
       
if __name__ == '__main__':
   app.run(debug = False) 
   
"""End of Yash's Contribution"""

# End.
#------------------------------------------------------------------------------------------------------------------   