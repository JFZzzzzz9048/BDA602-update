import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api

# from plotly import figure_factory as ff
# import seaborn as sns
from plotly import express as px
from plotly import graph_objects as go

# from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# from sklearn.inspection import permutation_importance


# Build a function to Determine if response is continuous or boolean (don't worry about >2 category responses)
def response_con_bool(response_list):
    if len(set(response_list)) == 2:
        return "boolean"
    else:
        return "continous"


# Determine if the predictor is cat/cont
def cont_bool(predictor_list):
    a = "categorical"
    b = "continous"
    if predictor_list.dtype == "category" or predictor_list.dtype == "object":
        return a
    elif len(set(predictor_list)) < 0.05 * len(predictor_list):
        return a
    else:
        return b


# print("Response '{}' is {}.".format(response[0], response_con_bool(df[response[0]])))

"""
cont_pred = []
for i in predictors:
    if cont_bool(df[i]) == "continous":
        cont_pred.append(i)
    print("Predictor '{}' is {}.".format(i, cont_bool(df[i])))
print(cont_pred)
"""


# Categorical Response by Categorical Predictor
def cat_response_cat_predictor(cat_response, cat_predictor):
    conf_matrix = confusion_matrix(cat_predictor.astype(str), cat_response.astype(str))

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file="cat_response_cat_predictor_heat_map_yes_relation.html",
        include_plotlyjs="cdn",
    )
    return


# cat_response_cat_predictor(df["survived"], df["sex"])


# Categorical Response by Continous Predictor
def cat_response_cont_predictor(cat_response, cont_predictor):
    # Group data together
    df = pd.DataFrame(
        {"predictor": cont_predictor.astype(float), "response": cat_response}
    )

    # Violin plot
    violin = px.violin(
        df, y="predictor", x="response", color="response", violinmode="overlay"
    )
    violin.update_layout(
        title="Violin plot of {} grouped by {}".format("predictor", "response")
    )
    violin.update_xaxes(title_text="response")
    violin.update_yaxes(title_text="predictor")
    violin.show()

    violin.write_html(
        file="cat_response_cont_predictor_violin_plot.html",
        include_plotlyjs="cdn",
    )

    # Distribution plot
    hist = px.histogram(
        df,
        x="response",
        y="predictor",
        color="response",
        marginal="box",
        hover_data=df.columns,
    )
    hist.update_layout(
        title="Histogram plot of {} grouped by {}".format("predictor", "response")
    )
    hist.show()

    hist.write_html(
        file="cat_response_cont_predictor_hist_plot.html",
        include_plotlyjs="cdn",
    )
    return


# cat_response_cont_predictor(df["survived"], df["age"])


# Continous Response by Categorical Predictor
def cont_response_cat_predictor(cont_response, cat_predictor):
    # Group data together
    df = pd.DataFrame(
        {"predictor": cat_predictor, "response": cont_response.astype(float)}
    )

    # Violin plot
    violin = px.violin(
        df, y="response", x="predictor", color="predictor", violinmode="overlay"
    )
    violin.update_layout(
        title="Violin plot of {} grouped by {}".format("response", "predictor")
    )
    violin.update_xaxes(title_text="predictor")
    violin.update_yaxes(title_text="response")
    violin.show()

    violin.write_html(
        file="cont_response_cat_predictor_violin_plot.html",
        include_plotlyjs="cdn",
    )

    # Distribution plot
    hist = px.histogram(
        df,
        x="predictor",
        y="response",
        color="predictor",
        marginal="box",
        hover_data=df.columns,
    )
    hist.update_layout(
        title="Histogram plot of {} grouped by {}".format("response", "predictor")
    )
    hist.show()

    hist.write_html(
        file="cont_response_cat_predictor_hist_plot.html",
        include_plotlyjs="cdn",
    )
    return


# cont_response_cat_predictor(df["age"], df["survived"])


# Continous Response by Continous Predictor
def cont_response_cont_predictor(cont_response, cont_predictor):
    # Group data together
    df = pd.DataFrame(
        {
            "predictor": cont_predictor.astype(float),
            "response": cont_response.astype(float),
        }
    )

    scatter = px.scatter(df, x="predictor", y="response", trendline="ols")
    scatter.update_layout(
        title_text="Scatter Plot: {} vs. {}".format("predictor", "response")
    )
    scatter.update_xaxes(ticks="inside", title_text="predictor")
    scatter.update_yaxes(ticks="inside", title_text="response")
    scatter.show()

    scatter.write_html(
        file="cont_response_cont_predictor_scatter_plot.html",
        include_plotlyjs="cdn",
    )
    return


# cont_response_cont_predictor(df["age"], df["body"])


# Continous Response by Continous Predictor Linear Regression
def plot_linear(cont_response, cont_predictor):
    y = cont_response.fillna(0).to_numpy()
    predictor = cont_predictor.fillna(0).to_numpy()

    predictor1 = statsmodels.api.add_constant(predictor)
    linear_regression_model = statsmodels.api.OLS(y, predictor1)
    linear_regression_model_fitted = linear_regression_model.fit()

    print(linear_regression_model_fitted.summary())

    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {'feature_name'}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {'feature_name'}",
        yaxis_title="y",
    )
    fig.show()

    fig.write_html(
        file="cont_response_cont_predictor_linear_regression_plot.html",
        include_plotlyjs="cdn",
    )
    return


# plot_linear(df["body"], df["age"])


# Categorical Response by Continous Predictor Logistic Regression
def plot_logistic(cat_response, cont_predictor):
    y = cat_response.astype(float).to_numpy()
    predictor = cont_predictor.fillna(0).to_numpy()

    predictor1 = statsmodels.api.add_constant(predictor)
    logistic_regression_model = statsmodels.api.GLM(y, predictor1)
    logistic_regression_model = logistic_regression_model.fit()

    print(logistic_regression_model.summary())

    t_value = round(logistic_regression_model.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {'feature_name'}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {'feature_name'}",
        yaxis_title="y",
    )
    fig.show()

    fig.write_html(
        file="cat_response_cont_predictor_logistic_regression_plot.html",
        include_plotlyjs="cdn",
    )
    return


# plot_logistic(df["survived"], df["age"])


# Random Forest Ranking and Plot
def random_forest_ranking(df_cont_pred, response):
    X = df_cont_pred.fillna(0)
    y = response
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()
    plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")
    important_df = pd.DataFrame(
        {
            "Rank": np.arange(1, len(df_cont_pred.columns) + 1, 1),
            "predictor": X.columns[-sorted_idx],
            "Importance": rf.feature_importances_[-sorted_idx],
        }
    )
    return important_df


# random_forest_ranking(df[cont_pred], df["survived"])


def main():
    # load dataset
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    # dataset only contains predictors
    df = titanic["data"]

    # add response variable
    df["survived"] = titanic["target"]

    # dataframe contains both a response and predictors
    df = pd.DataFrame(df)
    df.drop(
        ["boat", "body", "home.dest", "cabin"], axis=1, inplace=True
    )  # cabin has 77% missing value so just drop it
    df = df.dropna()

    # Given a list of predictors and the response columns
    response = ["survived"]
    predictors = list(df.columns)
    predictors.remove(response[0])
    predictors

    cont_pred = []
    if response_con_bool(df[response[0]]) == "continous":
        for i in predictors:
            if cont_bool(df[i]) == "continous":
                cont_pred.append(i)
                print(
                    "Continous Response by Continous Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cont_response_cont_predictor(df[response[0]], df[i])  # plot html
                plot_linear(
                    df[response[0]], df[i]
                )  # linear regression plot html, p-value, t-score
                random_forest_ranking(
                    df[cont_pred], df[response[0]]
                )  # Random Forest Ranking Dataframe and plot
            else:
                print(
                    "Continous Response by Categorical Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cont_response_cat_predictor(df[response[0]], df[i])  # plot html

    else:
        for i in predictors:
            if cont_bool(df[i]) == "continous":
                cont_pred.append(i)
                print(
                    "Categorical Response by Continous Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cat_response_cont_predictor(df[response[0]], df[i])  # plot html
                plot_logistic(
                    df[response[0]], df[i]
                )  # logictic regression plot html, p-value, t-score
                random_forest_ranking(
                    df[cont_pred], df[response[0]]
                )  # Random Forest Ranking Dataframe and plot
            else:
                print(
                    "Categorical Response by Categorical Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cat_response_cat_predictor(df[response[0]], df[i])  # plot html


if __name__ == "__main__":
    sys.exit(main())
