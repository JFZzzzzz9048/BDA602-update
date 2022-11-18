import os
import sys
import webbrowser
from itertools import combinations, product

import data_process

# import matplotlib
import pandas as pd
import plots
from cat_correlation import con_con_correlation
from plotly import graph_objects as go
from pyspark.sql import SparkSession
from sklearn.preprocessing import LabelEncoder

from hw5.difference_mean import (
    diff_mean_response,
    diff_mean_response_2d,
    random_forest_ranking,
)

# matplotlib.use("TkAgg")


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


createFolder("./plots/")


# Reference:
# https://www.geeksforgeeks.org/how-to-create-a-table-with-clickable-hyperlink-to-a-local-file-in-pandas/
def fun(path):
    # returns the final component of a url
    f_url = os.path.basename(path)
    # convert the url into link
    return '<a href="{}">{}</a>'.format(path, f_url)


def main():
    rootpath = os.getcwd()
    urlpath = f"file:///{rootpath}/plots/final_report.html"
    # print(urlpath)

    cont_pred = []
    cat_pred = []

    # Setup Spark
    database = "baseball"
    username = "root"
    pw = "password"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # SQL query to generate a batter info dataframe
    sql_baseball = """
        SELECT * FROM baseball_ready
        """

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    df_sql_baseball = (
        spark.read.format("jdbc")
        .options(
            url=jdbc_url,
            query=sql_baseball,
            user=username,
            password=pw,
            driver=jdbc_driver,
        )
        .load()
    )

    df_baseball = df_sql_baseball.toPandas()

    df_baseball.loc[(df_baseball.winner_home_or_away != "H"), "winner_home_or_away"] = 0
    df_baseball.loc[(df_baseball.winner_home_or_away == "H"), "winner_home_or_away"] = 1

    df_baseball.rename(columns={"winner_home_or_away": "HomeTeamWins"}, inplace=True)

    response = "HomeTeamWins"

    df_baseball_1 = df_baseball.drop(
        ["Game_ID", "home_team_id", "away_team_id", "game_date"], axis=1
    )
    predictors = list(df_baseball_1.columns)
    predictors.remove(response)

    # df, predictors, response = dataset_loader.get_test_data_set("titanic")
    df = data_process.clean_df(df_baseball_1)

    # Create continous predictors list & categorical predictors list
    for i in predictors:
        if data_process.cont_bool(df[i]) == "categorical":
            cat_pred.append(i)
        else:
            cont_pred.append(i)

    # print(cont_pred)

    # HW4 Table
    t_score = []
    p_value = []
    file_location_regression = []
    html_plot_file_location = []
    html_plot_file_location_dist = []
    predictor_name_df1 = []
    predictor_name_df2 = []
    predictor_name_df4 = []
    file_weight_link = []
    file_unweight_link = []
    file_location_diff_plot = []
    labelencoder = LabelEncoder()

    df[response] = labelencoder.fit_transform(df[response])
    for i in cont_pred:
        # print("Categorical Response by Continous Predictor: {} vs. {}".format(response[0], i))
        (
            predictor_name_1,
            file_location_cat_cont_vio,
            file_location_cat_cont_distribution,
        ) = plots.cat_response_cont_predictor(
            df[response], df[i], i, response
        )  # plot html
        (
            predictor_name_2,
            t_score_1,
            p_value_1,
            file_location_regression_1,
        ) = plots.plot_logistic(
            df[response], df[i], i, response
        )  # logictic regression plot html, p-value, t-score
        pred_out, importance = random_forest_ranking(df[cont_pred], df[response])
        (
            predictor_name_4,
            file_weight_1,
            file_unweight_1,
            file_location_diff_plot_1,
        ) = diff_mean_response(df[response], df[i], 10, i)

        # dataframe1
        predictor_name_df1.append(predictor_name_1)
        html_plot_file_location.append(file_location_cat_cont_vio)
        html_plot_file_location_dist.append(file_location_cat_cont_distribution)

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

    # Dataframe 1
    df1 = pd.DataFrame(
        {
            "PredictorName": predictor_name_df1,
            "Violin Plot": html_plot_file_location,
            "Histogram": html_plot_file_location_dist,
        }
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

    # Continous Predictors Table
    df_cont_pred = df1.copy()
    df_cont_pred.rename(columns={"PredictorName": "Continous Predictor"})

    # HW4 Table
    outer_merged_1 = pd.merge(df1, df2, how="outer", on=["PredictorName"])
    outer_merged_2 = pd.merge(df3, df4, how="outer", on=["PredictorName"])
    df_hw4_completed = pd.merge(
        outer_merged_1, outer_merged_2, how="outer", on=["PredictorName"]
    )
    df_hw4_completed.sort_values(by="Importance", ascending=False)

    # Continous & Continous
    predictors_column = []
    pearson_r = []
    absoulte_value_correlation = []
    cont_cont_linear_plot_link = []
    predictor1 = []
    predictor2 = []
    diff_mean = []
    weighted_diff_mean = []
    cont_cont_bin_link = []
    cont_cont_residual_link = []

    for cont_cont in combinations(cont_pred, 2):
        predictor_str = str(cont_cont[0]) + " and " + str(cont_cont[1])
        predictors_column.append(predictor_str)
        predictor1.append(cont_cont[0])
        predictor2.append(cont_cont[1])

        pearsonr = con_con_correlation(
            df[cont_cont[0]], cont_cont[0], df[cont_cont[1]], cont_cont[1]
        )
        pearson_r.append(pearsonr)
        absoulte_value_correlation.append(abs(pearsonr))

        # plot_linear(cont_response, cont_predictor, predictor_name, response_name):
        _, t_value, p_value, file_location = plots.plot_linear(
            df[cont_cont[0]], df[cont_cont[1]], cont_cont[0], cont_cont[1]
        )
        cont_cont_linear_plot_link.append(file_location)

        (
            diff_mean_value,
            weighted_diff_mean_value,
            cont_cont_bin,
            cont_cont_residual,
        ) = diff_mean_response_2d(
            df[cont_cont[0]],
            df[cont_cont[1]],
            df[response],
            5,
            cont_cont[0],
            cont_cont[1],
        )
        diff_mean.append(diff_mean_value)
        weighted_diff_mean.append(weighted_diff_mean_value)
        cont_cont_bin_link.append(cont_cont_bin)
        cont_cont_residual_link.append(cont_cont_residual)

    # Correlation Table:
    final_report_link = "plots/hw5_final_report.html"
    corr_df_link = []
    corr_df_predictor = []
    corr_df_predictor.append("Continous/Continous Correlation")
    cont_cont_corr_link = "cont_cont_corr_heatmap.html"
    corr_df_link.append(cont_cont_corr_link)
    cont_cont_correlation = pd.DataFrame(
        {
            "Predictors": predictors_column,
            "Pearson's r": pearson_r,
            "Absolute Value of Correlation": absoulte_value_correlation,
            "Linear Regression Plot": cont_cont_linear_plot_link,
        }
    )

    cont_cont_correlation = cont_cont_correlation.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )

    # Cont-Cont Correlation Matrix
    # Let us split this list into two parts
    cont_var1 = cont_pred
    cont_var2 = cont_pred

    # Creating all possible combinations between the above two variables list
    cont_var_prod = list(product(cont_var1, cont_var2, repeat=1))

    cont_cont_matrix = pd.DataFrame(
        {"Predictor1": predictor1, "Predictor2": predictor2, "Pearson R": pearson_r}
    )

    cont_var_prod = pd.DataFrame(cont_var_prod, columns=["Predictor1", "Predictor2"])
    cont_cont_final_matrix = pd.merge(
        cont_var_prod, cont_cont_matrix, how="left", on=["Predictor1", "Predictor2"]
    )
    cont_cont_final_matrix.loc[
        (cont_cont_final_matrix["Predictor1"] == cont_cont_final_matrix["Predictor2"]),
        "Pearson R",
    ] = 1

    # Using pivot function to convert the above DataFrame into a crosstab
    cont_cont_final_matrix = cont_cont_final_matrix.pivot(
        index="Predictor1", columns="Predictor2", values="Pearson R"
    )
    s2 = cont_cont_final_matrix.T
    cont_cont_final_matrix = cont_cont_final_matrix.fillna(s2)

    cont_cont_correlation_heatmap = go.Figure(
        data=go.Heatmap(
            z=cont_cont_final_matrix.to_numpy(),
            x=list(cont_cont_final_matrix.index),
            y=list(cont_cont_final_matrix.columns),
            hoverongaps=False,
            zmin=0,
            zmax=1,
        )
    )

    cont_cont_correlation_heatmap.update_layout(
        title="Categorical/Categorical Correlation Heatmap"
    )

    # cont_cont_correlation_heatmap.show()

    # Brute Force Table:
    cont_cont_brute = pd.DataFrame(
        {
            "Predictor 1": predictor1,
            "Predictor 2": predictor2,
            "Difference of Mean Response": diff_mean,
            "Weighted Difference of Mean Response": weighted_diff_mean,
            "Bin Plot": cont_cont_bin_link,
            "Residual Plot": cont_cont_residual_link,
        }
    )

    cont_cont_brute = cont_cont_brute.sort_values(
        by=["Weighted Difference of Mean Response"], ascending=False
    )
    # print(cont_cont_brute)

    df_hw4_completed = df_hw4_completed.style.format(
        {
            "Violin Plot": fun,
            "Histogram": fun,
            "RegressionPlot": fun,
            "MWR_plot_link": fun,
        }
    )

    df_cont_pred = df_cont_pred.style.format({"Violin Plot": fun, "Histogram": fun})

    cont_cont_correlation = cont_cont_correlation.style.format(
        {"Linear Regression Plot": fun}
    )

    cont_cont_brute = cont_cont_brute.style.format(
        {"Bin Plot": fun, "Residual Plot": fun}
    )

    with open(final_report_link, "w") as f:
        f.write(
            "--------Continous Predictors Table-------\n"
            + df_cont_pred.to_html()
            + "\n\n"
        )
        f.write("\n\n")
        f.write(
            "-------- Ranking Table(Importance is Decending)-------\n"
            + df_hw4_completed.to_html()
            + "\n\n"
        )
        f.write(
            "--------Correlation Matricies Table-------\n"
            + cont_cont_correlation_heatmap.to_html()
            + "\n\n"
        )
        f.write("\n\n")
        f.write(
            "\n-------Continous/Continous Correlation Table-------\n"
            + cont_cont_correlation.to_html()
            + "\n\n"
        )
        f.write("\n\n")
        f.write(
            "\n-------Continous/Continous Brute Table-------\n"
            + cont_cont_brute.to_html()
        )

        f.close()

    webbrowser.open(urlpath)


if __name__ == "__main__":
    sys.exit(main())
