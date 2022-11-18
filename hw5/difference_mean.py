import numpy as np
import pandas as pd
from plotly import graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()


# Difference with mean of response along with it's plot (unweighted)
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
def diff_mean_response(response_list, predictor_list, binNum, predictor_name):
    file_location = "{}_difference_mean_plot.html".format(predictor_name)

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

    bin_count = hist

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

    weight_mean_square_difference = df_weight["MeanSquaredDiffWeighted"].sum()
    mean_square_difference = df_unweight["MeanSquaredDiff"].mean()

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
        title=f"{predictor_name}",
        xaxis_title="Predictor Bin",
        yaxis_title="Response",
        yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
    )

    # fig_2.show()

    fig_2.write_html(
        file="plots/" + file_location,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    """
    html_weight = df_weight.to_html()
    html_unweight = df_unweight.to_html()
    text_file_weight = open(file_weight, "w")
    text_file_unweight = open(file_unweight, "w")
    text_file_weight.write(html_weight)
    text_file_unweight.write(html_unweight)
    text_file_weight.close()
    text_file_unweight.close()
    """

    return (
        predictor_name,
        weight_mean_square_difference,
        mean_square_difference,
        file_location,
    )


def diff_mean_response_2d(pred1, pred2, response, binNum, pred1_name, pred2_name):
    file_location_bin = "{}_{}_difference_mean_bin_plot.html".format(
        pred1_name, pred2_name
    )
    file_location_residual = "{}_{}_difference_mean_residual_plot.html".format(
        pred1_name, pred2_name
    )

    hist, x1bins, x2bins = np.histogram2d(pred1, pred2, bins=binNum)
    bin_means, binx_edges, biny_edges, binnumber = stats.binned_statistic_2d(
        pred1, pred2, response, statistic="mean", bins=[x1bins, x2bins], range=None
    )

    bin_count = hist

    population_mean = np.average(response)
    mean_sqr_diff = np.power((bin_means - population_mean), 2)
    population_prop = bin_count / len(response)
    mean_sqr_diff_weight = mean_sqr_diff * population_prop

    weight_mean_square_difference = np.nansum(mean_sqr_diff_weight)
    mean_square_difference = np.nanmean(mean_sqr_diff)

    z1 = bin_means
    z2 = bin_means - population_prop

    fig_bin = go.Figure(
        data=go.Heatmap(
            z=z1, x=x1bins, y=x2bins, hoverongaps=False, zmin=0, zmax=z1.max()
        )
    )

    fig_bin.update_layout(
        title=f"{pred1_name} by {pred2_name} Bin Plot",
        xaxis_title=pred1_name,
        yaxis_title=pred2_name,
    )

    fig_residual = go.Figure(
        data=go.Heatmap(
            z=z2, x=x1bins, y=x2bins, hoverongaps=False, zmin=0, zmax=z2.max()
        )
    )

    fig_residual.update_layout(
        title=f"{pred1_name} by {pred2_name} Residual Plot",
        xaxis_title=pred1_name,
        yaxis_title=pred2_name,
    )

    # fig_bin.show()
    # fig_residual.show()

    fig_bin.write_html(
        file="plots/" + file_location_bin,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    fig_residual.write_html(
        file="plots/" + file_location_residual,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    return (
        mean_square_difference,
        weight_mean_square_difference,
        file_location_bin,
        file_location_residual,
    )


# Random Forest and Ranking
def random_forest_ranking(df_cont_pred, response):
    X = df_cont_pred
    y = response
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=12
    )
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    sorted_idx = rf.feature_importances_.argsort()

    return X.columns[-sorted_idx], rf.feature_importances_[-sorted_idx]
