import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def model_df(X, y):

    # Train/Test Split (20%, 80%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)

    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    lr_results = pd.DataFrame(
        ["Linear regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]
    ).transpose()
    lr_results.columns = [
        "Method",
        "Training MSE",
        "Training R2",
        "Test MSE",
        "Test R2",
    ]

    # 2. Random Forest
    rf = RandomForestRegressor(max_depth=2, random_state=42)
    rf.fit(X_train, y_train)

    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)

    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    rf_results = pd.DataFrame(
        ["Random forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]
    ).transpose()
    rf_results.columns = [
        "Method",
        "Training MSE",
        "Training R2",
        "Test MSE",
        "Test R2",
    ]

    # 3. Navie Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    y_nb_train_pred = gnb.predict(X_train)
    y_nb_test_pred = gnb.predict(X_test)

    nb_train_mse = mean_squared_error(y_train, y_nb_train_pred)
    nb_train_r2 = r2_score(y_train, y_nb_train_pred)

    nb_test_mse = mean_squared_error(y_test, y_nb_test_pred)
    nb_test_r2 = r2_score(y_test, y_nb_test_pred)

    nb_results = pd.DataFrame(
        ["Navie Bayes", nb_train_mse, nb_train_r2, nb_test_mse, nb_test_r2]
    ).transpose()
    nb_results.columns = [
        "Method",
        "Training MSE",
        "Training R2",
        "Test MSE",
        "Test R2",
    ]

    naive_bayes = gnb.fit(X_train, y_train)
    nb_score = naive_bayes.score(X_test, y_test)
    print("Accuracy for Navice Bayes: {}".format(nb_score))

    linear_reg = lr.fit(X_train, y_train)
    lr_score = linear_reg.score(X_test, y_test)
    print("Accuracy for Linear Regression: {}".format(lr_score))

    ml_df = pd.concat([lr_results, rf_results, nb_results])

    # How to split train/test?
    # Basically, we need to select a date then split the data to part 1: before the date, part2: after the date.
    # Since my data set is order by game_id, from the oldest game to the lastest game. So, we can just split it by 20%,
    # 80%.

    # Which model is better?
    # Random Forest is better than linear regression because Random Forest has a higher R-squared and lower MSE.
    # It is strange, when I got Navies Bayes score is greater than Linear Regression. However, NB's R-Squared is
    # negative, which means the variance is extremely large. Overall, I would say Random Forest is the best among these
    # three models.

    return ml_df
