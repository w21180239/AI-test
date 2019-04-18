import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from keras.utils import to_categorical
from sklearn import preprocessing, metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ONLY_FLOAT = True
FULL_DATA = True
ONE_HOT = False


def load_data():
    print('Loading data...')
    df = pd.DataFrame(pd.read_csv('train.csv'))
    dd = pd.DataFrame(pd.read_csv('test.csv'))
    X = df.values
    X2 = np.vstack([df.values, dd.values])
    df2 = pd.DataFrame(pd.read_csv('label.csv'))
    class_list = [135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192,
                  195, 198, 201, 204, 209, 210, 247]
    float_list = [i for i in range(1, len(X[0])) if i not in class_list]
    class_fea = X[:, class_list].astype(np.str)
    full_class_fea = X2[:, class_list].astype(np.str)
    full_class_fea = np.delete(full_class_fea, 24, 1)
    class_fea = np.delete(class_fea, 24, 1)
    float_fea = X[:, float_list].astype(np.float32)
    cl_tran_list = [preprocessing.LabelEncoder() for i in range(len(class_fea[0]))]
    tmp_class_fea = []
    for i in range(len(cl_tran_list)):
        cl_tran_list[i].fit(full_class_fea[:, i])
    one = preprocessing.OneHotEncoder(n_values=[len(t.classes_) for t in cl_tran_list])
    for i in range(len(cl_tran_list)):
        tmp_class_fea.append(cl_tran_list[i].transform(class_fea[:, i]))
    class_fea = np.array(tmp_class_fea).astype(np.float32).transpose((1, 0))
    if ONE_HOT:
        class_fea = one.fit_transform(class_fea)
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=1)
    float_fea = imp.fit_transform(float_fea)

    label = df2.values[:, 1]
    label = to_categorical(label)
    if ONLY_FLOAT:
        feature = float_fea
    else:
        if ONE_HOT:
            class_fea = class_fea.astype(np.float32).toarray()
        feature = np.hstack([float_fea, class_fea])
    ss = preprocessing.MaxAbsScaler().fit(feature)
    feature = ss.transform(feature)
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.1, random_state=100)
    if FULL_DATA:
        x_train = feature
        y_train = label
    count = 0
    for i in range(len(y_train)):
        if y_train[i][1] == 1:
            count += 1
    scale = (len(y_train) - count) / count
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=1 - 1e-3, random_state=200)
    print('Finished!')
    return x_train, x_test, y_train, y_test, x_val, y_val, ss, cl_tran_list, imp, scale, one


def write_out(y_pred):
    print('Writing to file...')
    idd = np.array([i for i in range(0, y_pred.size)])
    idd = np.reshape(idd, (idd.size, 1))
    y_pred = np.reshape(y_pred, (y_pred.size, 1))
    re = np.concatenate([idd, y_pred], axis=1)
    tmp = pd.DataFrame(re, columns=['Id', 'label'], dtype=float)
    tmp[['Id']] = tmp[['Id']].astype(int)
    tmp.to_csv('result.csv', index=False)
    print("writing file finished!")


def Classify(x_train, x_test, y_train, y_test, ss, cl_tran_list, imp, scale, one):
    y_train = np.argmax(y_train, 1)
    y_test = np.argmax(y_test, 1)
    parameters = {'n_jobs': [1], 'n_estimatores': [50, 100, 200, 300], 'learning_rate': [0.5, 1],
                  'min_child_weight': [200, 300, 400], 'max_depth': [4, 5, 6, 7], 'gamma': [0.1],
                  'reg_lambda': [4], 'scale_pos_weight': [scale], 'tree_method': ['gpu_hist'],
                  'predictor': ['gpu_predictor']}
    classifier = XGBClassifier(n_estimators=400, learning_rate=0.5, min_child_weight=300, max_depth=6, gamma=0.1,
                              reg_lambda=2,
                              scale_pos_weight=scale, tree_method='gpu_hist', predictor='gpu_predictor')
    # grid_classifier = GridSearchCV(n_jobs=4, param_grid=parameters, estimator=classifier, scoring='auc', verbose=1)
    grid_classifier = classifier
    # grid_classifier.fit(x_train, y_train)
    print('Starting fitting...')
    t1 = time.time()
    classifier.fit(x_train, y_train)
    print('Time:%.4fs' % (time.time() - t1))

    # cv_result = pd.DataFrame.from_dict(grid_classifier.cv_results_)
    # with open('cv_result_3.csv', 'w') as f:
    #     cv_result.to_csv(f)
    # print('The parameters of the best model are: ')
    # print(grid_classifier.best_params_)
    # fig, ax = plt.subplots(figsize=(10, 15))
    # plot_importance(classifier, height=0.5, max_num_features=64, ax=ax)
    # plt.show()
    classifier_y_predit = grid_classifier.predict_proba(x_test)[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, classifier_y_predit)
    print('AUC of Classifier:%f' % metrics.auc(fpr, tpr))
    # print(classification_report(y_test, classifier_y_predit))
    print('Predicating...')
    df = pd.DataFrame(pd.read_csv('test.csv'))
    X = df.values
    class_list = [135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192,
                  195, 198, 201, 204, 209, 210, 247]
    float_list = [i for i in range(1, len(X[0])) if i not in class_list]
    class_fea = X[:, class_list].astype(np.str)
    class_fea = np.delete(class_fea, 24, 1)

    float_fea = X[:, float_list].astype(np.float32)
    tmp_class_fea = []

    float_fea = imp.transform(float_fea)

    if ONLY_FLOAT:
        feature = float_fea
    else:
        for i in range(len(cl_tran_list)):
            tmp_class_fea.append(cl_tran_list[i].transform(class_fea[:, i]))
        class_fea = np.array(tmp_class_fea).astype(np.float32).transpose((1, 0))
        if ONE_HOT:
            class_fea = one.fit_transform(class_fea).astype(np.float32).toarray()
        feature = np.hstack((class_fea, float_fea))
    feature = ss.transform(feature)
    y = grid_classifier.predict_proba(feature)[:,1]
    print('Predication finished!')
    return y


x_train, x_test, y_train, y_test, x_val, y_val, ss, cl_tran_list, imp, scale, one = load_data()
y = Classify(x_train, x_test, y_train, y_test, ss, cl_tran_list, imp, scale, one)
write_out(y)
