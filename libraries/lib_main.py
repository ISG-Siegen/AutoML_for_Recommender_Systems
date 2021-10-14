import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from auto_scikit_learn import automl
from data_processing.preprocessing import preprocessing_1m, preprocessing_100k
from scikit_learn import random_forest
from scikit_learn import knn
from scikit_learn import sgd
from scikit_learn import svr
from H2O import H2O

pd.set_option('display.max_columns', None)

#___________________________________________________Configuration_______________________________________________________
# Set to False if estimation should not run
sklearn_random_forest_config = True
sklearn_knn_config = True
sklearn_svr_config = True
sklearn_sgd_config = False
sklearn_automl_config = False
h2o_config = False

#ONLY ONE DATASET CONFIG SHOULD BE TRUE!
movielens_100k_config = True
movielens_1m_config = False

pd.set_option('display.max_columns', None)
#___________________________________________________Data_Statistics____________________________________________________

if movielens_100k_config == True:
    rm_df, features, label, num_users, num_movies = preprocessing_100k.preprocess_ml_100k()

if movielens_1m_config == True:
    rm_df, features, label, num_users, num_movies = preprocessing_1m.preprocess_ml_1m()

# print("Number of Users :" + str(num_users))
# print("Number of Movies :" + str(num_movies))
# print(rm_df.head())
#
# #print(rm_df['rating'].value_counts())
# rm_df['rating'].value_counts().plot(kind='bar')
# plt.savefig('ratings.png')


#___________________________________________________Split_Dataset_______________________________________________________



x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.25)

print('xtrain :' + str(x_train.shape))
print('xtest :' + str(x_test.shape))
print('ytrain :' + str(y_train.shape))
print('ytest :' + str(y_test.shape))

#___________________________________________________EXECUTION___________________________________________________________
random_forest_rmse = 0
knn_rmse = 0
sgd_rmse = 0
svr_rmse = 0
automl_rmse = 0
h2o_rmse = 0


if sklearn_random_forest_config == True:
    print('START RANDOM FOREST')
    random_forest_rmse, random_forest_execTime = random_forest.random_forest(x_train, x_test, y_train, y_test)
    print('RANDOM FOREST FINISHED')

if sklearn_knn_config == True:
    print('START KNN')
    knn_rmse, knn_execTime = knn.knn(x_train, x_test, y_train, y_test)
    print('KNN FINISHED')

if sklearn_sgd_config == True:
    print('START SGD')
    sgd_rmse, sgd_execTime = sgd.sgd(x_train, x_test, y_train, y_test)
    print('KNN FINISHED')

if sklearn_svr_config == True:
    print('START SVR')
    svr_rmse, svr_execTime = svr.svr(x_train, x_test, y_train, y_test)
    print('SVR FINISHED')

if sklearn_automl_config == True:
    print('START AUOSKLEARN')
    automl_rmse, automl_execTime = automl.automl(x_train, x_test, y_train, y_test)
    print('AUOSKLEARN FINISHED')

if h2o_config == True:
    print('START H2O')
    h2o_rmse, h2o_execTime = H2O.H2O(rm_df)
    print('H2O FINISHED')

#___________________________________________________OUTPUT______________________________________________________________
if sklearn_random_forest_config == True:
    print('Training Time Random Forest: '+ str(random_forest_execTime) + ' Sekunden')
    print('Random Forest RMSE : ' + str(random_forest_rmse))

if sklearn_knn_config == True:
    print('Training Time KNN: '+ str(knn_execTime) + ' Sekunden')
    print('KNN RMSE : ' + str(knn_rmse))

if sklearn_sgd_config == True:
    print('Training Time SGD: '+ str(sgd_execTime) + ' Sekunden')
    print('SGD RMSE : ' + str(sgd_rmse))

if sklearn_svr_config == True:
    print('Training Time SVR: '+ str(svr_execTime) + ' Sekunden')
    print('SVR RMSE : ' + str(svr_rmse))

if sklearn_automl_config == True:
    print('Training Time AutoMl: ' + str(automl_execTime) + ' Sekunden')
    print('autoMl RMSE : ' + str(automl_rmse))

if h2o_config == True:
    print('Training Time H2O: ' + str(h2o_execTime) + ' Sekunden')
    print('H2O RMSE : ' + str(h2o_rmse))



#___________________________________________________PLOT_RESULTS________________________________________________________

height = [random_forest_rmse, knn_rmse, svr_rmse, 1.120, h2o_rmse, 0.936]
bars = ('RF - scikit', 'KNN - scikit', 'SVR - scikit', 'ASR - autokslearn', 'GBE - H2O', 'S - Surprise')
x_pos = np.arange(len(bars))

plt.bar(x_pos, height)
plt.xticks(x_pos, bars)
plt.ylim([0.8, 1.2])
plt.savefig('endresult.png')
plt.show()


