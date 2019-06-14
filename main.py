import os
import time
import warnings

import lightgbm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ONLY_FLOAT = False
FULL_DATA = True
ONE_HOT = True


def process_class_fea(full_class_fea, class_fea):
    cl_tran_list = [preprocessing.LabelEncoder() for i in range(len(class_fea[0]))]
    tmp_class_fea = []
    for i in range(len(cl_tran_list)):
        cl_tran_list[i].fit(full_class_fea[:, i])
    final_class_list = [cl_tran_list.index(t) for t in cl_tran_list if len(t.classes_) < 6]
    class_fea = class_fea[:, final_class_list]
    cl_tran_list = cl_tran_list[final_class_list]
    one = preprocessing.OneHotEncoder(n_values=[len(t.classes_) for t in cl_tran_list])
    for i in range(len(cl_tran_list)):
        tmp_class_fea.append(cl_tran_list[i].transform(class_fea[:, i]))
    class_fea = np.array(tmp_class_fea).astype(np.float32).transpose((1, 0))
    if ONE_HOT:
        class_fea = one.fit_transform(class_fea)
    return class_fea, cl_tran_list, one, final_class_list


def load_data():
    print('Loading data...')
    feature = []
    label = []
    for i in range(1, 6):
        feature.append(pd.DataFrame(pd.read_csv(f'train{i}.csv', header=None)).values)
        label.append(pd.DataFrame(pd.read_csv(f'label{i}.csv', header=None)).values)
    feature = np.concatenate(feature, 0)
    label = np.concatenate(label, 0)
    ss = preprocessing.MaxAbsScaler().fit(feature)
    feature = ss.transform(feature)
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, random_state=100)
    x_train = feature
    y_train = label

    print('Finished!')
    return x_train, x_test, y_train, y_test, ss


def write_out(y_pred):
    print('Writing to file...')
    idd = np.array([i for i in range(1, y_pred.size + 1)])
    idd = np.reshape(idd, (idd.size, 1))
    y_pred = np.reshape(y_pred, (y_pred.size, 1))
    re = np.concatenate([idd, y_pred], axis=1)
    tmp = pd.DataFrame(re, columns=['Id', 'Predicted'], dtype=float)
    tmp[['Id']] = tmp[['Id']].astype(int)
    tmp.to_csv('result.csv', index=False)
    print("writing file finished!")


def Classify(x_train, x_test, y_train, y_test, ss):
    grid = False

    classifier = XGBRegressor(n_estimators=50, learning_rate=0.5, min_child_weight=400, max_depth=4, gamma=0.1,
                              reg_lambda=4, tree_method='gpu_hist', predictor='gpu_predictor')
    # classifier = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor')
    classifier = lightgbm.LGBMRegressor(n_estimators=1000)
    if grid:
        parameters = {'n_jobs': [-1], 'n_estimatores': [50, 100, 200, 300], 'learning_rate': [0.5, 1],
                      'min_child_weight': [200, 300, 400], 'max_depth': [4, 5, 6, 7], 'gamma': [0.1],
                      'reg_lambda': [4], 'tree_method': ['gpu_hist'],
                      'predictor': ['gpu_predictor']}
        grid_classifier = GridSearchCV(n_jobs=1, param_grid=parameters, estimator=classifier, scoring='r2', verbose=1)
    else:
        grid_classifier = classifier
    print('Starting fitting...')
    t1 = time.time()
    grid_classifier.fit(x_train, y_train)

    # classifier.fit(x_train, y_train)
    print('Time:%.4fs' % (time.time() - t1))

    if grid:
        cv_result = pd.DataFrame.from_dict(grid_classifier.cv_results_)
        with open('cv_result_3.csv', 'w') as f:
            cv_result.to_csv(f)
        print('The parameters of the best model are: ')
        print(grid_classifier.best_params_)
    # fig, ax = plt.subplots(figsize=(10, 15))
    # plot_importance(classifier, height=0.5, max_num_features=64, ax=ax)
    # plt.show()

    y_predict = grid_classifier.predict(x_test)

    print(r2_score(y_test, y_predict))
    print('Predicating...')
    del x_train, y_train, x_test, y_test, y_predict
    feature = []
    for i in range(1, 7):
        feature.append(pd.DataFrame(pd.read_csv(f'test{i}.csv', header=None)).values)
    feature = np.concatenate(feature, 0)
    feature = ss.transform(feature)
    y = grid_classifier.predict(feature)
    print('Predication finished!')
    return y


x_train, x_test, y_train, y_test, ss = load_data()
y = Classify(x_train, x_test, y_train, y_test, ss)
write_out(y)