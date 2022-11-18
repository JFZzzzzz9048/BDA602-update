import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor


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

    # 3. Extra Tree
    et = ExtraTreeRegressor(random_state=42)
    et.fit(X_train, y_train)

    y_et_train_pred = et.predict(X_train)
    y_et_test_pred = et.predict(X_test)

    et_train_mse = mean_squared_error(y_train, y_et_train_pred)
    et_train_r2 = r2_score(y_train, y_et_train_pred)
    et_test_mse = mean_squared_error(y_test, y_et_test_pred)
    et_test_r2 = r2_score(y_test, y_et_test_pred)

    et_results = pd.DataFrame(
        ["Extra Tree", et_train_mse, et_train_r2, et_test_mse, et_test_r2]
    ).transpose()
    et_results.columns = [
        "Method",
        "Training MSE",
        "Training R2",
        "Test MSE",
        "Test R2",
    ]

    ml_df = pd.concat([lr_results, rf_results, et_results])

    return ml_df
