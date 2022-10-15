import os

# my def main() also not work, so i just gave up use main()
import sys
import webbrowser

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import graph_objects as go
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


createFolder("./plots/")
createFolder("./dataframes/")
"""

# rootpath = "/Users/jingfei/Dropbox/Mac (2)/Documents/GitHub/BDA_HW_plot/plots"
rootpath = os.path.dirname(sys.argv[0])
urlpath = "https://jfzzzzzz9048.github.io/BDA_HW_plot/plots"


# build a function to clean dataframe
def clean_df(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == "category" or dataframe[i].dtype == "object":
            # categorical datatype fillna with mode
            num = dataframe[i].mode()[0]
            dataframe[i].fillna(num, inplace=True)
            # Drop categorical predictor with more than half different categories
            if len(set(dataframe[i])) > 0.5 * len(dataframe[i]):
                dataframe = dataframe.drop(columns=i)
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

    file_location = "%s/categorical_%s_categorical_%s_heatmap_plot.html" % (
        rootpath,
        response_name,
        predictor_name,
    )

    url_location = "%s/categorical_%s_categorical_%s_heatmap_plot.html" % (
        urlpath,
        response_name,
        predictor_name,
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

    return "{}_Categorical".format(predictor_name), url_location


# Build a function to plot Categorical Response by Continous Predictor
def cat_response_cont_predictor(
    cat_response, cont_predictor, predictor_name, response_name
):

    file_location_cat_cont = "%s/categorical_%s_continous_%s_violin_hist_plot.html" % (
        rootpath,
        response_name,
        predictor_name,
    )

    url_location = "%s/categorical_%s_continous_%s_violin_hist_plot.html" % (
        urlpath,
        response_name,
        predictor_name,
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
    # violin.show()

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
    # hist.show()

    with open(file_location_cat_cont, "a") as f:
        f.write(violin.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(hist.to_html(full_html=False, include_plotlyjs="cdn"))

    return "{}_Continous".format(predictor_name), url_location


# Build a function to plot Continous Response by Categorical Predictor
def cont_response_cat_predictor(
    cont_response, cat_predictor, predictor_name, response_name
):

    file_location_cont_cat = "%s/continous_%s_categorical_%s_violin_hist_plot.html" % (
        rootpath,
        response_name,
        predictor_name,
    )

    url_location = "%s/continous_%s_categorical_%s_violin_hist_plot.html" % (
        urlpath,
        response_name,
        predictor_name,
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
    # violin.show()

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
    # hist.show()

    with open(file_location_cont_cat, "a") as f:
        f.write(violin.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(hist.to_html(full_html=False, include_plotlyjs="cdn"))

    return "{}_Categorical".format(predictor_name), url_location


# Build a function to plot Continous Response by Continous Predictor
def cont_response_cont_predictor(
    cont_response, cont_predictor, predictor_name, response_name
):

    file_location = "%s/continous_%s_countinous_%s_plot.html" % (
        rootpath,
        response_name,
        predictor_name,
    )

    url_location = "%s/continous_%s_countinous_%s_plot.html" % (
        urlpath,
        response_name,
        predictor_name,
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
    # scatter.show()

    scatter.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return "{}_Continous".format(predictor_name), url_location


# Perform a linear Regression and return p-value and t-score
def plot_linear(cont_response, cont_predictor, predictor_name, response_name):

    file_location = "%s/%s_linear_regression_plot.html" % (rootpath, predictor_name)

    url_location = "%s/%s_linear_regression_plot.html" % (urlpath, predictor_name)

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
    # fig.show()

    fig.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return "{}_Continous".format(predictor_name), t_value, p_value, url_location


# Perform a logistic regression and return p-value and t-score
def plot_logistic(cat_response, cont_predictor, predictor_name, response_name):

    file_location = "%s/%s_logistic_regression_plot.html" % (rootpath, predictor_name)

    url_location = "%s/%s_logistic_regression_plot.html" % (urlpath, predictor_name)

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
    # fig.show()

    fig.write_html(
        file=file_location,
        include_plotlyjs="cdn",
    )
    return "{}_Continous".format(predictor_name), t_value, p_value, url_location


# Random Forest and Ranking
def random_forest_ranking(df_cont_pred, response):
    X = df_cont_pred
    y = response
    output = []
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()
    # plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    for i in X.columns[-sorted_idx]:
        output.append(i + "_Continous")
    # print(output)
    return output, rf.feature_importances_[-sorted_idx]


# Difference with mean of response along with it's plot (unweighted)
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
def diff_mean_response(response_list, predictor_list, binNum, predictor_name):
    labelencoder = LabelEncoder()
    if predictor_name in cont_pred:
        predictor_name = predictor_name + "_Continous"
    else:
        predictor_name = predictor_name + "_Categorical"

    file_location = "%s/%s_difference_mean_plot.html" % (rootpath, predictor_name)
    url_location = "%s/%s_difference_mean_plot.html" % (urlpath, predictor_name)

    file_weight = "%s/weight_%s_mean_diff_table.html" % (rootpath, predictor_name)
    url_weight_location = "%s/weight_%s_mean_diff_table.html" % (
        urlpath,
        predictor_name,
    )

    file_unweight = "%s/unweight_%s_mean_diff_table.html" % (rootpath, predictor_name)
    url_unweight_location = "%s/unweight_%s_mean_diff_table.html" % (
        urlpath,
        predictor_name,
    )

    hist, bins = np.histogram(labelencoder.fit_transform(predictor_list), binNum)
    bin_means, bin_edges, binnumber = stats.binned_statistic(
        labelencoder.fit_transform(predictor_list),
        labelencoder.fit_transform(response_list),
        statistic="mean",
        bins=binNum,
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2
    population_mean = np.average(labelencoder.fit_transform(response_list))

    lower_bound = bin_centers - 0.5 * bin_width
    upper_bound = bin_centers + 0.5 * bin_width
    bin_centers
    bin_count = hist
    bin_means
    population_mean_1 = [population_mean] * binNum
    mean_sqr_diff = np.power((bin_means - population_mean), 2)
    population_prop = bin_count / len(response_list)
    mean_sqr_diff_weight = mean_sqr_diff * population_prop

    d = {
        "LowerBound": lower_bound,
        "UpperBound": upper_bound,
        "BinCenter": bin_centers,
        "BinCount": bin_count,
        "BinMean": bin_means,
        "PopulationMean": population_mean_1,
        "MeanSquaredDiff": mean_sqr_diff,
        "PopulationPorption": population_prop,
        "MeanSquaredDiffWeighted": mean_sqr_diff_weight,
    }

    df_weight = pd.DataFrame(data=d)

    df_unweight = df_weight.drop(
        columns=["PopulationPorption", "MeanSquaredDiffWeighted"]
    )

    fig_2 = go.Figure()

    fig_2.add_trace(
        go.Bar(x=bin_centers, y=hist, yaxis="y2", name="Population", opacity=0.5)
    )

    # Bin Mean - Population
    fig_2.add_trace(
        go.Scatter(x=bin_centers, y=bin_means, name="\u03BC_i - \u03BC_Population")
    )

    # Population Mean
    fig_2.add_trace(
        go.Scatter(
            x=[
                np.min(labelencoder.fit_transform(predictor_list)),
                np.max(labelencoder.fit_transform(predictor_list)),
            ],
            y=[population_mean, population_mean],
            name="\u03BC Population",
            mode="lines",
        )
    )

    fig_2.update_layout(
        title=f"Difference with mean of response - {predictor_name}",
        xaxis_title="Predictor Bin",
        yaxis_title="Response",
        yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
    )

    # fig_2.show()

    fig_2.write_html(
        file=file_location,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    html_weight = df_weight.to_html()
    html_unweight = df_unweight.to_html()
    text_file_weight = open(file_weight, "w")
    text_file_unweight = open(file_unweight, "w")
    text_file_weight.write(html_weight)
    text_file_unweight.write(html_unweight)
    text_file_weight.close()
    text_file_unweight.close()

    return predictor_name, url_weight_location, url_unweight_location, url_location


# load dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)
# dataset only contains predictors
df = titanic["data"]

# add response variable
df["survived"] = titanic["target"]

# dataframe contains both a response and predictors
df = pd.DataFrame(df)

# Call clean_df() function to clean data
df = clean_df(df)

# Given a list of predictors and the response columns
response = ["survived"]
predictors = list(df.columns)
predictors.remove(response[0])

cont_pred = []
t_score = []
p_value = []
file_location_regression = []
html_plot_file_location = []
predictor_name_df1 = []
predictor_name_df2 = []
predictor_name_df4 = []
file_weight_link = []
file_unweight_link = []
file_location_diff_plot = []
labelencoder = LabelEncoder()


if response_con_bool(df[response[0]]) == "continous":
    for i in predictors:
        if cont_bool(df[i]) == "continous":
            cont_pred.append(i)
            # print("Continous Response by Continous Predictor: {} vs. {}".format(response[0], i))
            predictor_name_1, file_location_cont_cont_1 = cont_response_cont_predictor(
                df[response[0]], df[i], i, response[0]
            )  # plot html
            (
                predictor_name_2,
                t_score_1,
                p_value_1,
                file_location_regression_1,
            ) = plot_linear(
                df[response[0]], df[i], i, response[0]
            )  # linear regression plot html, p-value, t-score
            pred_out, importance = random_forest_ranking(
                df[cont_pred], df[response[0]]
            )  # Random Forest Ranking Dataframe and plot
            (
                predictor_name_4,
                file_weight_1,
                file_unweight_1,
                file_location_diff_plot_1,
            ) = diff_mean_response(df[response[0]], df[i], 10, i)

            # dataframe1
            predictor_name_df1.append(predictor_name_1)
            html_plot_file_location.append(file_location_cont_cont_1)

            # dataframe2
            predictor_name_df2.append(predictor_name_2)
            t_score.append(t_score_1)
            p_value.append(p_value_1)
            file_location_regression.append(file_location_regression_1)

            # dataframe4
            predictor_name_df4.append(predictor_name_4)
            file_weight_link.append(file_weight_1)
            file_unweight_link.append(file_unweight_1)
            file_location_diff_plot.append(file_location_diff_plot_1)

        else:
            df[i] = labelencoder.fit_transform(df[i])
            # print("Continous Response by Categorical Predictor: {} vs. {}".format(response[0], i))

            predictor_name_1, file_location_cont_cont_1 = cont_response_cat_predictor(
                df[response[0]], df[i], i, response[0]
            )
            (
                predictor_name_4,
                file_weight_1,
                file_unweight_1,
                file_location_diff_plot_1,
            ) = diff_mean_response(df[response[0]], df[i], len(set(df[i])), i)

            # dataframe1
            predictor_name_df1.append(predictor_name_1)
            html_plot_file_location.append(file_location_cont_cont_1)

            # dataframe4
            predictor_name_df4.append(predictor_name_4)
            file_weight_link.append(file_weight_1)
            file_unweight_link.append(file_unweight_1)
            file_location_diff_plot.append(file_location_diff_plot_1)


else:
    df[response[0]] = labelencoder.fit_transform(df[response[0]])
    for i in predictors:
        if cont_bool(df[i]) == "continous":
            cont_pred.append(i)
            # print("Categorical Response by Continous Predictor: {} vs. {}".format(response[0], i))
            predictor_name_1, file_location_cat_cont_1 = cat_response_cont_predictor(
                df[response[0]], df[i], i, response[0]
            )  # plot html
            (
                predictor_name_2,
                t_score_1,
                p_value_1,
                file_location_regression_1,
            ) = plot_logistic(
                df[response[0]], df[i], i, response[0]
            )  # logictic regression plot html, p-value, t-score
            pred_out, importance = random_forest_ranking(df[cont_pred], df[response[0]])
            (
                predictor_name_4,
                file_weight_1,
                file_unweight_1,
                file_location_diff_plot_1,
            ) = diff_mean_response(df[response[0]], df[i], 10, i)

            # dataframe1
            predictor_name_df1.append(predictor_name_1)
            html_plot_file_location.append(file_location_cat_cont_1)

            # dataframe2
            predictor_name_df2.append(predictor_name_2)
            t_score.append(t_score_1)
            p_value.append(p_value_1)
            file_location_regression.append(file_location_regression_1)

            # dataframe4
            predictor_name_df4.append(predictor_name_4)
            file_weight_link.append(file_weight_1)
            file_unweight_link.append(file_unweight_1)
            file_location_diff_plot.append(file_location_diff_plot_1)

        else:
            df[i] = labelencoder.fit_transform(df[i])
            # print("Categorical Response by Categorical Predictor: {} vs. {}".format(response[0], i))
            predictor_name_1, file_location_cat_cat_1 = cat_response_cat_predictor(
                df[response[0]], df[i], i, response[0]
            )  # plot html
            (
                predictor_name_4,
                file_weight_1,
                file_unweight_1,
                file_location_diff_plot_1,
            ) = diff_mean_response(df[response[0]], df[i], len(set(df[i])), i)

            # dataframe1
            predictor_name_df1.append(predictor_name_1)
            html_plot_file_location.append(file_location_cat_cat_1)

            # dataframe4
            predictor_name_df4.append(predictor_name_4)
            file_weight_link.append(file_weight_1)
            file_unweight_link.append(file_unweight_1)
            file_location_diff_plot.append(file_location_diff_plot_1)

# Dataframe 1
df1 = pd.DataFrame(
    {"PredictorName": predictor_name_df1, "Plot": html_plot_file_location}
)

# Dataframe 2
df2 = pd.DataFrame(
    {
        "PredictorName": predictor_name_df2,
        "RegressionPlot": file_location_regression,
        "p-value": p_value,
        "t-score": t_score,
    }
)

# Dataframe 3
df3 = pd.DataFrame({"PredictorName": pred_out, "Importance": importance})

# Dataframe 4
df4 = pd.DataFrame(
    {
        "PredictorName": predictor_name_df4,
        "MWR_Unweighted": file_unweight_link,
        "MWR_Weighted": file_weight_link,
        "MWR_plot_link": file_location_diff_plot,
    }
)

outer_merged_1 = pd.merge(df1, df2, how="outer", on=["PredictorName"])
outer_merged_2 = pd.merge(df3, df4, how="outer", on=["PredictorName"])
df_completed = pd.merge(
    outer_merged_1, outer_merged_2, how="outer", on=["PredictorName"]
)
print(df_completed)

final_report_url = "%s/report.html" % (urlpath)
final_report_file = "%s/report.html" % (rootpath)

df_completed.to_html(
    final_report_file,
    classes="table table-striped",
    escape=False,
    render_links=True,
)
print(final_report_url)
webbrowser.open(final_report_url)


# My final report website is: https://jfzzzzzz9048.github.io/BDA_HW_plot/plots/report.html
