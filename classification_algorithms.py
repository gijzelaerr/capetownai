from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import tree

import numpy as np
import matplotlib.pyplot as plt


def classification_ranf(X, y):
    X_learn, X_test, y_learn, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.25)

    # A good rule of thumb when tuning the max_features parameter of the hyperparameter
    # algorithm is to set it equal to the squareroot of the total number of features. As
    # such, during tuning the hyperparameter is varied from about 30% less than this rule
    # of thumb value, to about 30% above it.
    min_features = round((len(X.columns) ** 0.5) * 0.7)
    max_features = round((len(X.columns) ** 0.5) * 1.3)

    parameters = {'n_estimators': [100, 300, 400], #number of trees in the forest
                  'max_features': [i for i in range(min_features, max_features, 1)],
                  'criterion': ['gini', 'entropy']} #metric according to which purity of split is determined

    model = RandomForestClassifier()

    classes = list(set(y))

    if len(classes) > 2:  #determine whether the task is a binary or multilabel classification task
        for k in list(parameters.keys()):
            parameters['estimator__' + k] = parameters.pop(k)   #the onevsallclassifier wants the parameters to be preceeded by the label 'esitimator__'
        model = OneVsRestClassifier(model)
        y_learn = label_binarize(y_learn, classes=classes)  #label_binarize is similar to onehotencoding
        y_test = label_binarize(y_test, classes=classes)

    grid_search = GridSearchCV(model, parameters, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1,
                               return_train_score=True)
    best_model = grid_search.fit(X_learn, y_learn)  #using gridsearch to tune hyperparameters

    print('best parameters:', best_model.best_params_)
    print('performance on training data:', best_model.best_score_)
    y_pred = best_model.predict(X_test)
    print('performance on test data:', roc_auc_score(y_test, y_pred))

    # investigating the extent to which each feature contributed to the prediction task
    model = RandomForestClassifier(criterion='entropy', max_features=5, n_estimators=400)
    best_model = model.fit(X_learn, y_learn)
    y_pred = best_model.predict(X_test)
    print('performance on test data:', roc_auc_score(y_test, y_pred))
    tat = best_model.feature_importances_

    # visualising vector coefficients
    colors = (0, 0, 0)
    area = np.pi * 3
    headings = np.arange(len(X.columns))
    plt.scatter(headings, tat, s=area, c=colors, alpha=0.5)
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('features')
    plt.ylabel('coefficient')
    plt.show()



def classification_decistree(X, y):
    X_learn, X_test, y_learn, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.25)

    parameters = {'criterion': ['gini', 'entropy'],
                  'min_samples_leaf': list(range(1, 500, 1))}

    model = DecisionTreeClassifier()

    classes = list(set(y))

    if len(classes) > 2:
        for k in list(parameters.keys()):
            parameters['estimator__' + k] = parameters.pop(k)
        model = OneVsRestClassifier(model)
        y_learn = label_binarize(y_learn, classes=classes)
        y_test = label_binarize(y_test, classes=classes)

    grid_search = GridSearchCV(model, parameters, cv=10, scoring='roc_auc', n_jobs=-1, verbose=1,
                               return_train_score=True)
    best_model = grid_search.fit(X_learn, y_learn)


    print('best parameters:', best_model.best_params_)
    print('performance on training data:', best_model.best_score_)
    y_pred = best_model.predict(X_test)
    print('performance on test data:', roc_auc_score(y_test, y_pred))

    print('Please select the splitting criterion:')
    criterion = str(input().strip())
    print('Please select min_samples_leaf:')
    min_samples_leaf = int(input().strip())

    model = DecisionTreeClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf)
    best_model = model.fit(X_learn, y_learn)

    plot_tree(best_model, max_depth=4, filled=True, fontsize= 9)
    plt.show()