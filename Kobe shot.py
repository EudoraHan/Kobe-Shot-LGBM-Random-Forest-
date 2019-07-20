
"""
Created on Wed Jul 17 16:47:08 2019

KOBE SHOT Evaluation and Prediction

@author: Yun Han
"""
# Random Forest Regression & LGBM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================================================================
# # Part 1: Data Exploration
# =============================================================================
# 1.1 Read data:
data = pd.read_csv('kobe_shot.csv')
data.shape # (30697, 25)

# 1.2: Variable Selection
# 1. Remove the row without target variable: shot_made_flag
data = data[pd.notnull(data['shot_made_flag'])]
data.shape # (25697, 25)

# 2. Compare the variable lat,lon & loc_x,loc_y. 
### Histogram
sns.distplot(data['loc_y'])
sns.distplot(data['lat'])
sns.distplot(data['loc_x'])
sns.distplot(data['lon'])

### Distribution Plot
plt.figure(figsize = (10,10))
 
plt.subplot(1,2,1)
plt.scatter(data.loc_x,data.loc_y,color ='r',alpha = 0.05)
plt.title('loc_x and loc_y')
 
plt.subplot(1,2,2)
plt.scatter(data.lon,data.lat,color ='b',alpha = 0.05)
plt.title('lat and lon')

### Correlations Matrix and heatmap
corr = data[["lat", "lon", "loc_x", "loc_y"]].corr()
sns.heatmap(corr)

"""
Both the distribution plot and the correlation matrix showed 
the lat,lon & loc_x,loc_y represent the same position. We can delete one of them.
"""


# 3. Time remain: 
"""
The variable 'minutes_remaining' and 'seconds_remaining' contain the same information，
we can combine the two variable together.

"""
data['remain_time'] = data['minutes_remaining'] * 60 + data['seconds_remaining']

# 4. shot_distance and shot_zone_range
corr = data[["shot_distance ", "shot_zone_range"]].corr()
sns.heatmap(corr)



# =============================================================================
# # Part 2. Feature Preprocessing:
# =============================================================================

# Delete the duplicated variable:
data = data.drop(['lat','lon','minutes_remaining', 'seconds_remaining','matchup',
                  'shot_id', 'team_id','team_name', 'shot_zone_range','game_date'], axis = 1)

# Use auto-data preprocessing fucntion
cat = ['action_type', 'combined_shot_type', 'period', 'playoffs','season', 'shot_made_flag',
       'shot_type', 'shot_zone_area', 'shot_zone_basic', 'opponent']        
after_Clean = clean_all(data, cat, 'delete', 'median')        

corr = after_Clean[['loc_x_x', 'loc_y_x','action_type_x', 'combined_shot_type_x', 'period_x', 'playoffs_x',
             'season_x', 'shot_made_flag_x','shot_type_x', 'shot_zone_area_x', 'shot_zone_basic_x',
             'opponent_x','game_event_id_x','game_id_x','shot_distance_x', 'remain_time_x']].corr()
sns.heatmap(corr)

"""
From the correlation matrix, we find that the correlation between shot_distance and loc_y is 0.79,
game_event_id and period is 0.96, game_id and playoffs is 0.92, hence, we can delete one of each pair.

"""

# Delete the high correlation variable: shot_distance, game_event_id,game_id
shot = after_Clean.drop(['shot_distance_x','game_event_id_x','game_id_x'], axis = 1)


# =============================================================================
# # Part 3: Model Training --- Random Forest
# =============================================================================
#X = shot.iloc[:, shot.columns!=  'shot_made_flag_x' ].values
X = shot.drop(['shot_made_flag_x'], axis = 1)
y = shot.iloc[:, 5].values


# One HotEncoder: Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1,2,3,5,7,10,11])
X = onehotencoder.fit_transform(X).toarray()


# 3.1: Split dataset
# Splite data into training and testing
from sklearn import model_selection
# Reserve 20% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,random_state = 0)

# 3.2: Model Training and Selection
from sklearn.ensemble import RandomForestClassifier
# Random Forest
classifier_RF = RandomForestClassifier()

# Train the model
classifier_RF.fit(X_train, y_train)
# Prediction of test data
classifier_RF.predict(X_test)

# Accuracy of test data
classifier_RF.score(X_test, y_test) #0.6369

# Use 5-fold Cross Validation to get the accuracy # 0.6192
cv_score = model_selection.cross_val_score(classifier_RF, X_train, y_train, cv=5)
print('Model accuracy of Random Forest is: %.3f',cv_score.mean())

# 3.3: Use Grid Search to Find Optimal Hyperparameters
from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results 
def print_grid_search_metrics(gs):
    print ("Best score: %0.3f" % gs.best_score_)
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Possible hyperparamter options for Random Forest
# Choose the number of trees
parameters = {
    'n_estimators' : [80,90]
}
Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)


# best number of tress
print_grid_search_metrics(Grid_RF)

"""
Best score: 0.641
Best parameters set:
        n_estimators: 80
"""

# best random forest
best_RF_model = Grid_RF.best_estimator_

# =============================================================================
# # Part 3-2: Model Evaluation - Confusion Matrix (Precision, Recall, Accuracy)
# =============================================================================




# =============================================================================
# # Part 3-3: Feature Selection:  check feature importance
# =============================================================================
# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(X, y)

importances = forest.feature_importances_

# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
for k,v in sorted(zip(map(lambda x: round(x, 4), importances), X.columns), reverse=True):
    print (v + ": " + str(k))




# =============================================================================
# # Part 4: Model Training --- LGBM
# =============================================================================
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=50,
    learning_rate=0.03,
    num_leaves=10,
    colsample_bytree=.8,
    subsample=.9,
    max_depth=7,
    reg_alpha=.1,
    reg_lambda=.1,
    min_split_gain=.01,
    min_child_weight=2,
    silent=-1,
    verbose=-1,
)

lgbm.fit(X_train, y_train, 
    eval_set= [(X_train, y_train), (X_test, y_test)], 
    eval_metric='auc', verbose=100, early_stopping_rounds=10  #30
)

# Use 5-fold Cross Validation to get the accuracy # 0.6192
cv_score = model_selection.cross_val_score(lgbm, X_train, y_train, cv=5)
print('Model accuracy of LGBM is:',cv_score.mean())

### Part 4.2: Use Grid Search to Find Optimal Hyperparameters
# Choose the number of trees
parameters = {
    'n_estimators' : [60, 80, 100]
}
Grid_RF = GridSearchCV(LGBMClassifier(),parameters, cv=5)
Grid_RF.fit(X_train, y_train)

# best number of tress
print_grid_search_metrics(Grid_RF)





# =============================================================================
# print(data['shot_type'].unique())
# print(data['shot_zone_range'].unique())
# print(data['shot_zone_basic'].unique())
# print(data['shot_zone_area'].unique())
#  
# print(data['shot_type'].value_counts())
# 
# print(data['season'].unique())
#  
# data['season'] = data['season'].apply(lambda x: int(x.split('-')[1]) )
# data['season'].unique()
# 
# 
# data['season']
# 
# import matplotlib.cm as cm
# plt.figure(figsize=(20,10))
#  
#  
# def scatterbygroupby(feature):
#     alpha = 0.1
#     gb = data.groupby(feature)
#     cl = cm.rainbow(np.linspace(0,1,len(gb)))
#     for g,c in zip(gb,cl):
#         plt.scatter(g[1].loc_x,g[1].loc_y,color = c,alpha = alpha) #这里为什么是g[1]还没搞清楚，希望知道的可以告知一下，谢谢!
#  
# plt.subplot(1,3,1)
# scatterbygroupby('shot_zone_basic')
# plt.title('shot_zone_basic')
#  
# plt.subplot(1,3,2)
# scatterbygroupby('shot_zone_range')
# plt.title('shot_zone_range')
#  
# plt.subplot(1,3,3)
# scatterbygroupby('shot_zone_area')
# plt.title('shot_zone_area')
# =============================================================================
