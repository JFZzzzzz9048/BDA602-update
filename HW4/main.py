import sys

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

# import seaborn as sns
import statsmodels.api

# from plotly import figure_factory as ff
from plotly import express as px
from plotly import graph_objects as go

# from scipy import stats
# from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor

# from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Build a function to clean dataframe, return a ready to use dataframe
def clean_df(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == "category" or dataframe[i].dtype == "object":
            # categorical datatype fillna with mode
            num = dataframe[i].mode()[0]
            dataframe[i].fillna(num, inplace=True)
            # dataframe[i] = labelencoder.fit_transform(dataframe[i])

        elif len(set(dataframe[i])) < 0.05 * len(dataframe[i]):
            # categorical datatype fillna with mode
            num = dataframe[i].mode()[0]
            dataframe[i].fillna(num, inplace=True)
            # dataframe[i] = labelencoder.fit_transform(dataframe[i])
        else:
            # numerical datatype fillna with mean
            num = dataframe[i].mean()
            dataframe[i].fillna(num, inplace=True)
    return dataframe


# Build a function to Determine if response is continuous or boolean (don't worry about >2 category responses)
def response_con_bool(response_list):
    if len(set(response_list)) == 2:
        return "boolean"
    else:
        return "continous"


# Determine if the predictor is cat/cont
def cont_bool(predictor_list):
    if predictor_list.dtype == "category" or predictor_list.dtype == "object":
        return "categorical"
    elif len(set(predictor_list)) < 0.05 * len(predictor_list):
        return "categorical"
    else:
        return "continous"


# Build a function to plot Categorical Response by Categorical Predictor
def cat_response_cat_predictor(
    cat_response, cat_predictor, predictor_name, response_name
):
    file_location = "plots/categorical_{}_categorical_{}_heatmap_plot.html".format(
        response_name, predictor_name
    )

    conf_matrix = confusion_matrix(cat_predictor.astype(str), cat_response.astype(str))

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title=f"Categorical Predictor {predictor_name} by Categorical Response {response_name} (with relationship)",
        xaxis_title=response_name,
        yaxis_title=predictor_name,
        # yaxis_range=[1.5,3.5],
        # xaxis_range=[-0.5,1.5]
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return predictor_name, file_location


# Build a function to plot Categorical Response by Continous Predictor
def cat_response_cont_predictor(
    cat_response, cont_predictor, predictor_name, response_name
):
    file_location_violin = "plots/categorical_{}_continous_{}_violin_plot.html".format(
        response_name, predictor_name
    )
    file_location_hist = "plots/categorical_{}_continous_{}_hist_plot.html".format(
        response_name, predictor_name
    )

    # Group data together
    df = pd.DataFrame(
        {"predictor": cont_predictor.astype(float), "response": cat_response}
    )

    # Violin plot
    violin = px.violin(
        df, y="predictor", x="response", color="response", violinmode="overlay"
    )
    violin.update_layout(
        title="Violin plot of {} grouped by {}".format(predictor_name, response_name)
    )
    violin.update_xaxes(title_text=response_name)
    violin.update_yaxes(title_text=predictor_name)
    violin.show()

    violin.write_html(
        file=file_location_violin,
        include_plotlyjs="cdn",
    )

    # Distribution plot
    hist = px.histogram(
        df,
        x="predictor",
        y="predictor",
        color="response",
        marginal="box",
        hover_data=df.columns,
    )
    hist.update_layout(
        title="Histogram plot of {} grouped by {}".format(predictor_name, response_name)
    )
    hist.show()

    hist.write_html(
        file=file_location_hist,
        include_plotlyjs="cdn",
    )
    return predictor_name, file_location_violin, file_location_hist


# Build a function to plot Continous Response by Categorical Predictor
def cont_response_cat_predictor(
    cont_response, cat_predictor, predictor_name, response_name
):
    file_location_violin = "plots/continous_{}_categorical_{}_violin_plot.html".format(
        response_name, predictor_name
    )
    file_location_hist = "plots/continous_{}_categorical_{}_hist_plot.html".format(
        response_name, predictor_name
    )
    # Group data together
    df = pd.DataFrame(
        {"predictor": cat_predictor, "response": cont_response.astype(float)}
    )

    # Violin plot
    violin = px.violin(
        df, y="response", x="predictor", color="predictor", violinmode="overlay"
    )
    violin.update_layout(
        title="Violin plot of {} grouped by {}".format(response_name, predictor_name)
    )
    violin.update_xaxes(title_text=predictor_name)
    violin.update_yaxes(title_text=response_name)
    violin.show()

    violin.write_html(
        file=file_location_violin,
        include_plotlyjs="cdn",
    )

    # Distribution plot
    hist = px.histogram(
        df,
        x="response",
        y="response",
        color="predictor",
        marginal="box",
        hover_data=df.columns,
    )
    hist.update_layout(
        title="Histogram plot of {} grouped by {}".format(response_name, predictor_name)
    )
    hist.show()

    hist.write_html(
        file=file_location_hist,
        include_plotlyjs="cdn",
    )
    return predictor_name, file_location_violin, file_location_hist


# Build a function to plot Continous Response by Continous Predictor
def cont_response_cont_predictor(
    cont_response, cont_predictor, predictor_name, response_name
):
    file_location = "plots/continous_{}_countinous_{}_plot.html".format(
        response_name, predictor_name
    )

    # Group data together
    df = pd.DataFrame(
        {
            "predictor": cont_predictor.astype(float),
            "response": cont_response.astype(float),
        }
    )

    scatter = px.scatter(df, x="predictor", y="response", trendline="ols")
    scatter.update_layout(
        title_text="Scatter Plot: {} vs. {}".format(predictor_name, response_name)
    )
    scatter.update_xaxes(ticks="inside", title_text=predictor_name)
    scatter.update_yaxes(ticks="inside", title_text=response_name)
    scatter.show()

    scatter.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return predictor_name, file_location


# Plot linear Regression and calculate p-value and t-score
def plot_linear(cont_response, cont_predictor, predictor_name, response_name):
    file_location = "plots/{}_linear_regression_plot.html".format(predictor_name)
    y = cont_response.fillna(0).to_numpy()
    predictor = cont_predictor.fillna(0).to_numpy()

    predictor1 = statsmodels.api.add_constant(predictor)
    linear_regression_model = statsmodels.api.OLS(y, predictor1)
    linear_regression_model_fitted = linear_regression_model.fit()

    # print(linear_regression_model_fitted.summary())

    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title="Variable: {}".format(predictor_name),
        yaxis_title=response_name,
    )
    fig.show()

    fig.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return t_value, p_value, file_location


# Plot logistic Regression and calculate p-value and t-score
def plot_logistic(cat_response, cont_predictor, predictor_name, response_name):
    file_location = "plots/{}_logistic_regression_plot.html".format(predictor_name)

    y = cat_response.astype(float).to_numpy()
    predictor = cont_predictor.fillna(0).to_numpy()

    predictor1 = statsmodels.api.add_constant(predictor)
    logistic_regression_model = statsmodels.api.Logit(y, predictor1)
    logistic_regression_model = logistic_regression_model.fit()

    # print(logistic_regression_model.summary())

    t_value = round(logistic_regression_model.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title="Variable: {}".format(predictor_name),
        yaxis_title=response_name,
    )
    fig.show()

    fig.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return t_value, p_value, file_location


# Perform Random Forest and Ranking
def random_forest_ranking(df_cont_pred, response):
    X = df_cont_pred
    y = response
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()
    # plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    """
    important_df = pd.DataFrame(
        {
            "Rank": np.arange(1, len(df_cont_pred.columns) + 1, 1),
            "Predictor": X.columns[-sorted_idx],
            "Importance": rf.feature_importances_[-sorted_idx],
        }
    )
    """
    return X.columns[-sorted_idx], rf.feature_importances_[-sorted_idx]


def main():
    # load dataset
    titanic = fetch_openml("titanic", version=1, as_frame=True)
    # dataset only contains predictors
    df = titanic["data"]

    # add response variable
    df["survived"] = titanic["target"]

    # dataframe contains both a response and predictors
    df = pd.DataFrame(df)
    df = clean_df(df)

    # Given a list of predictors and the response columns
    response = ["survived"]
    predictors = list(df.columns)
    predictors.remove(response[0])

    cont_pred = []
    t_score = []
    p_value = []
    # rank = []
    # file_location = []
    labelencoder = LabelEncoder()

    if response_con_bool(df[response[0]]) == "continous":
        for i in predictors:
            if cont_bool(df[i]) == "continous":
                cont_pred.append(i)
                print(
                    "Continous Response by Continous Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cont_response_cont_predictor(
                    df[response[0]], df[i], i, response[0]
                )  # plot html
                t_score_1, p_value_1 = plot_linear(
                    df[response[0]], df[i], i, response[0]
                )  # linear regression plot html, p-value, t-score
                t_score.append(t_score_1)
                p_value.append(p_value_1)
                pred_out, importance = random_forest_ranking(
                    df[cont_pred], df[response[0]]
                )  # Random Forest Ranking Dataframe and plot

            else:
                df[i] = labelencoder.fit_transform(df[i])
                print(
                    "Continous Response by Categorical Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cont_response_cat_predictor(
                    df[response[0]], df[i], i, response[0]
                )  # plot html

    else:
        df[response[0]] = labelencoder.fit_transform(df[response[0]])
        for i in predictors:
            if cont_bool(df[i]) == "continous":
                cont_pred.append(i)
                print(
                    "Categorical Response by Continous Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cat_response_cont_predictor(
                    df[response[0]], df[i], i, response[0]
                )  # plot html
                t_score_1, p_value_1, file_location = plot_logistic(
                    df[response[0]], df[i], i, response[0]
                )  # logictic regression plot html, p-value, t-score
                t_score.append(t_score_1)
                p_value.append(p_value_1)
                pred_out, importance = random_forest_ranking(
                    df[cont_pred], df[response[0]]
                )  # Random Forest Ranking Dataframe and plot

            else:
                print(
                    "Categorical Response by Categorical Predictor: {} vs. {}".format(
                        response[0], i
                    )
                )
                cat_response_cat_predictor(
                    df[response[0]], df[i], i, response[0]
                )  # plot html


if __name__ == "__main__":
    sys.exit(main())
