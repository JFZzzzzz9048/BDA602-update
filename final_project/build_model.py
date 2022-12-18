# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def model_df(X, y):
    # Train/Test Split (30%, 70%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_lr_test_pred = lr.predict(X_test)

    fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(
        y_test, y_lr_test_pred, pos_label=1
    )
    auc_lr = round(metrics.roc_auc_score(y_test, y_lr_test_pred), 4)
    print("AUC for Logistic Regression: {}".format(auc_lr))

    lr_results = pd.DataFrame(["Logistic Regression", auc_lr]).transpose()
    lr_results.columns = [
        "Method",
        "AUC",
    ]

    # 2. Random Forest
    rf = RandomForestRegressor(max_depth=3, min_samples_leaf=10, random_state=42)
    rf.fit(X_train, y_train)

    y_rf_test_pred = rf.predict(X_test)

    fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(
        y_test, y_rf_test_pred, pos_label=1
    )
    auc_rf = round(metrics.roc_auc_score(y_test, y_rf_test_pred), 4)
    print("AUC for Random Forest: {}".format(auc_rf))

    rf_results = pd.DataFrame(["Random Forest", auc_rf]).transpose()
    rf_results.columns = [
        "Method",
        "AUC",
    ]

    # 3. Navie Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_nb_test_pred = gnb.predict(X_test)

    fpr_nb, tpr_nb, thresholds_nb = metrics.roc_curve(
        y_test, y_nb_test_pred, pos_label=1
    )
    auc_nb = round(metrics.roc_auc_score(y_test, y_nb_test_pred), 4)
    print("AUC for Navie Bayes: {}".format(auc_nb))

    nb_results = pd.DataFrame(["Navie Bayes", auc_nb]).transpose()
    nb_results.columns = [
        "Method",
        "AUC",
    ]

    # 4. Decision Tree
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_clf_pred = clf.predict(X_test)
    fpr_clf, tpr_clf, thresholds_clf = metrics.roc_curve(
        y_test, y_clf_pred, pos_label=1
    )
    auc_clf = round(metrics.roc_auc_score(y_test, y_clf_pred), 4)

    print("AUC for Decision Tree: {}".format(auc_clf))

    clf_results = pd.DataFrame(["Decision Tree", auc_clf]).transpose()
    clf_results.columns = [
        "Method",
        "AUC",
    ]

    ml_df = pd.concat([lr_results, rf_results, nb_results, clf_results])
    ml_df.to_csv("model_score.csv")

    plt.figure(0).clf()
    plt.plot(fpr_lr, tpr_lr, label="Logistic Regression, AUC=" + str(auc_lr))
    plt.plot(fpr_rf, tpr_rf, label="Random Forest, AUC=" + str(auc_rf))
    plt.plot(fpr_nb, tpr_nb, label="Naive Bayes, AUC=" + str(auc_nb))
    plt.plot(fpr_clf, tpr_clf, label="Decision Tree, AUC=" + str(auc_clf))

    plt.title("ROC Curves")
    # add legend
    plt.legend()
    plt.savefig("ROC.pdf")

    return ml_df
