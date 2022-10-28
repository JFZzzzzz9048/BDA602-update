import os
import webbrowser
from itertools import combinations, product

# import sys
import data_process
import dataset_loader
import matplotlib
import numpy as np
import pandas as pd
import plots
from cat_correlation import (
    cat_cont_correlation_ratio,
    cat_correlation,
    con_con_correlation,
)
from plotly import graph_objects as go
from scipy import stats
from sklearn.preprocessing import LabelEncoder

matplotlib.use("TkAgg")


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


createFolder("./plots/")


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

    fig_bin.show()
    fig_residual.show()

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


# Reference:
# https://www.geeksforgeeks.org/how-to-create-a-table-with-clickable-hyperlink-to-a-local-file-in-pandas/
def fun(path):
    # returns the final component of a url
    f_url = os.path.basename(path)
    # convert the url into link
    return '<a href="{}">{}</a>'.format(path, f_url)


"""
def main():



if __name__ == "__main__":
    sys.exit(main())
"""

rootpath = os.getcwd()
urlpath = f"file:///{rootpath}/plots/final_report.html"
# print(urlpath)

cont_pred = []
cat_pred = []

df, predictors, response = dataset_loader.get_test_data_set("titanic")
df = data_process.clean_df(df)

# Create continous predictors list & categorical predictors list
for i in predictors:
    if data_process.cont_bool(df[i]) == "categorical":
        cat_pred.append(i)
    else:
        cont_pred.append(i)


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
        df[cont_cont[0]], df[cont_cont[1]], df[response], 5, cont_cont[0], cont_cont[1]
    )
    diff_mean.append(diff_mean_value)
    weighted_diff_mean.append(weighted_diff_mean_value)
    cont_cont_bin_link.append(cont_cont_bin)
    cont_cont_residual_link.append(cont_cont_residual)


# Correlation Table:
final_report_link = "plots/final_report.html"
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

cont_cont_correlation_heatmap = go.Figure(
    data=go.Heatmap(
        z=df[cont_pred].corr(method="pearson").to_numpy(),
        x=list(set(cont_pred)),
        y=list(set(cont_pred)),
        hoverongaps=False,
        zmin=0,
        zmax=df[cont_pred].corr(method="pearson").to_numpy().max(),
    )
)

cont_cont_correlation_heatmap.update_layout(
    title="Categorical/Categorical Correlation Heatmap"
)

# cont_cont_correlation_heatmap.show()

cont_cont_correlation_heatmap.write_html(
    file="plots/" + cont_cont_corr_link, include_plotlyjs="cdn"
)


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


# Categorical & Categorical
cat_predictors_column = []
cramers_v = []
cat_cat_absoulte = []
cat_cat_heatmap_link = []
cat_predictor1 = []
cat_predictor2 = []
cat_cat_diff_mean = []
cat_cat_weighted_diff_mean = []
cat_cat_bin_link = []
cat_cat_redidual_link = []
labelencoder = LabelEncoder()

for cat_cat in combinations(cat_pred, 2):
    cat_predictor_str = str(cat_cat[0]) + " and " + str(cat_cat[1])
    cat_predictors_column.append(cat_predictor_str)
    cat_predictor1.append(cat_cat[0])
    cat_predictor2.append(cat_cat[1])

    cramersv = cat_correlation(df[cat_cat[0]], cat_cat[0], df[cat_cat[1]], cat_cat[1])
    cramers_v.append(cramersv)
    cat_cat_absoulte.append(abs(cramersv))

    _, cat_file_location = plots.cat_response_cat_predictor(
        df[cat_cat[0]], df[cat_cat[1]], cat_cat[0], cat_cat[1]
    )
    cat_cat_heatmap_link.append(cat_file_location)

    (
        cat_cat_diff_mean_value,
        cat_cat_weighted_diff_mean_value,
        cat_cat_bin,
        cat_cat_residual,
    ) = diff_mean_response_2d(
        labelencoder.fit_transform(df[cat_cat[0]]),
        labelencoder.fit_transform(df[cat_cat[1]]),
        df[response],
        5,
        cat_cat[0],
        cat_cat[1],
    )
    cat_cat_diff_mean.append(cat_cat_diff_mean_value)
    cat_cat_weighted_diff_mean.append(cat_cat_weighted_diff_mean_value)
    cat_cat_bin_link.append(cat_cat_bin)
    cat_cat_redidual_link.append(cat_cat_residual)


# Correlation Table:
cat_cat_correlation = pd.DataFrame(
    {
        "Predictors": cat_predictors_column,
        "Cramer's V": cramers_v,
        "Absolute Value of Correlation": cat_cat_absoulte,
        "Heatmap": cat_cat_heatmap_link,
    }
)

cat_cat_correlation = cat_cat_correlation.sort_values(
    by=["Absolute Value of Correlation"], ascending=False
)
# print(cat_cat_correlation)


# Correlation Matrix
df_cat_v1 = df[cat_pred]
# Let us split this list into two parts
cat_var1 = cat_pred
cat_var2 = cat_pred

# Creating all possible combinations between the above two variables list
cat_var_prod = list(product(cat_var1, cat_var2, repeat=1))

cat_cat_matrix = pd.DataFrame(
    {
        "Predictor1": cat_predictor1,
        "Predictor2": cat_predictor2,
        "Cramer's V": cramers_v,
    }
)

cat_var_prod = pd.DataFrame(cat_var_prod, columns=["Predictor1", "Predictor2"])
cat_cat_final_matrix = pd.merge(
    cat_var_prod, cat_cat_matrix, how="left", on=["Predictor1", "Predictor2"]
)
cat_cat_final_matrix.loc[
    (cat_cat_final_matrix["Predictor1"] == cat_cat_final_matrix["Predictor2"]),
    "Cramer's V",
] = 1

# Using pivot function to convert the above DataFrame into a crosstab
cat_cat_final_matrix = cat_cat_final_matrix.pivot(
    index="Predictor1", columns="Predictor2", values="Cramer's V"
)
s2 = cat_cat_final_matrix.T
cat_cat_final_matrix = cat_cat_final_matrix.fillna(s2)
# print(cat_cat_final_matrix)

# Correlation Plot
corr_df_predictor.append("Categorical/Categorical Correlation")
cat_cat_corr_link = "cat_cat_corr_heatmap.html"
corr_df_link.append(cat_cat_corr_link)

cat_cat_correlation_heatmap = go.Figure(
    data=go.Heatmap(
        z=cat_cat_final_matrix.to_numpy(),
        x=list(set(cat_predictor1)),
        y=list(set(cat_predictor1)),
        hoverongaps=False,
        zmin=0,
        zmax=cat_cat_final_matrix.to_numpy().max(),
    )
)

cat_cat_correlation_heatmap.update_layout(
    title="Categorical/Categorical Correlation Heatmap"
)

cat_cat_correlation_heatmap.show()

cat_cat_correlation_heatmap.write_html(
    file="plots/" + cat_cat_corr_link, include_plotlyjs="cdn"
)

# Brute Force Table:
cat_cat_brute = pd.DataFrame(
    {
        "Predictor 1": cat_predictor1,
        "Predictor 2": cat_predictor2,
        "Difference of Mean Response": cat_cat_diff_mean,
        "Weighted Difference of Mean Response": cat_cat_weighted_diff_mean,
        "Bin Plot": cat_cat_bin_link,
        "Residual Plot": cat_cat_redidual_link,
    }
)

cat_cat_brute = cat_cat_brute.sort_values(
    by=["Weighted Difference of Mean Response"], ascending=False
)

# print(cat_cat_brute)


# Continous & Categorical
cont_cat_predictors_column = []
correlation_ratio = []
cont_cat_absoulte = []
cont_cat_violin_link = []
cont_cat_dist_link = []
cont_predictor1 = []
cat_predictor2 = []
cont_cat_diff_mean = []
cont_cat_weighted_diff_mean = []
cont_cat_bin_link = []
cont_cat_residual_link = []
labelencoder = LabelEncoder()

for cont_predictor in cont_pred:
    for cat_predictor in cat_pred:
        cont_cat_predictor_str = str(cont_predictor) + " and " + str(cat_predictor)
        cont_cat_predictors_column.append(cont_cat_predictor_str)
        cont_predictor1.append(cont_predictor)
        cat_predictor2.append(cat_predictor)
        correlation_ratio_num = cat_cont_correlation_ratio(
            df[cat_predictor], np.array(df[cont_predictor])
        )
        correlation_ratio.append(correlation_ratio_num)
        cont_cat_absoulte.append(abs(correlation_ratio_num))

        (
            _,
            file_location_cat_cont_violin,
            file_location_cat_cont_dist,
        ) = plots.cat_response_cont_predictor(
            df[cat_predictor], df[cont_predictor], cat_predictor, cont_predictor
        )
        cont_cat_violin_link.append(file_location_cat_cont_violin)
        cont_cat_dist_link.append(file_location_cat_cont_dist)

        (
            cont_cat_diff_mean_value,
            cont_cat_weighted_diff_mean_value,
            cont_cat_bin,
            cont_cat_residual,
        ) = diff_mean_response_2d(
            df[cont_predictor],
            labelencoder.fit_transform(df[cat_predictor]),
            df[response],
            5,
            cont_predictor,
            cat_predictor,
        )
        cont_cat_diff_mean.append(cont_cat_diff_mean_value)
        cont_cat_weighted_diff_mean.append(cont_cat_weighted_diff_mean_value)

        cont_cat_bin_link.append(cont_cat_bin)
        cont_cat_residual_link.append(cont_cat_residual)

cont_cat_correlation = pd.DataFrame(
    {
        "Predictors": cont_cat_predictors_column,
        "Correlation Ratio": correlation_ratio,
        "Absolute Value of Correlation": cont_cat_absoulte,
        "Violin Plot": cont_cat_violin_link,
        "Distribution Plot": cont_cat_dist_link,
    }
)

cont_cat_correlation = cont_cat_correlation.sort_values(
    by=["Absolute Value of Correlation"], ascending=False
)
# print(cont_cat_correlation)


# Correlation Matrix
cont_cat_matrix = pd.DataFrame(
    {
        "Predictor1": cont_predictor1,
        "Predictor2": cat_predictor2,
        "Correlation Ratio": correlation_ratio,
    }
)

cont_cat_matrix = cont_cat_matrix.pivot(
    index="Predictor1", columns="Predictor2", values="Correlation Ratio"
)

# print(cont_cat_matrix)


# Correlation Plot
corr_df_predictor.append("Continous/Categorical Correlation")
cont_cat_corr_link = "cont_cat_corr_heatmap.html"
corr_df_link.append(cont_cat_corr_link)

cont_cat_correlation_heatmap = go.Figure(
    data=go.Heatmap(
        z=cont_cat_matrix.to_numpy(),
        x=list(set(cat_predictor2)),
        y=list(set(cont_predictor1)),
        hoverongaps=False,
        zmin=0,
        zmax=cont_cat_matrix.to_numpy().max(),
    )
)

cont_cat_correlation_heatmap.update_layout(
    title="Continous/Categorical Correlation Heatmap"
)

cont_cat_correlation_heatmap.show()

cont_cat_correlation_heatmap.write_html(
    file="plots/" + cont_cat_corr_link, include_plotlyjs="cdn"
)

# Brute Force Table:
cont_cat_brute = pd.DataFrame(
    {
        "Predictor 1": cont_predictor1,
        "Predictor 2": cat_predictor2,
        "Difference of Mean Response": cont_cat_diff_mean,
        "Weighted Difference of Mean Response": cont_cat_weighted_diff_mean,
        "Bin Plot": cont_cat_bin_link,
        "Residual Plot": cont_cat_residual_link,
    }
)

cont_cat_brute = cont_cat_brute.sort_values(
    by=["Weighted Difference of Mean Response"], ascending=False
)

# print(cont_cat_brute)

df_correlation_matricies = pd.DataFrame(
    {
        "Predictor1/Predictor2": corr_df_predictor,
        "Correlation Heatmap Link": corr_df_link,
    }
)


df_correlation_matricies = df_correlation_matricies.style.format(
    {"Correlation Heatmap Link": fun}
)
cont_cont_correlation = cont_cont_correlation.style.format(
    {"Linear Regression Plot": fun}
)
cat_cat_correlation = cat_cat_correlation.style.format({"Heatmap": fun})
cont_cat_correlation = cont_cat_correlation.style.format(
    {"Violin Plot": fun, "Distribution Plot": fun}
)
cont_cont_brute = cont_cont_brute.style.format({"Bin Plot": fun, "Residual Plot": fun})
cat_cat_brute = cat_cat_brute.style.format({"Bin Plot": fun, "Residual Plot": fun})
cont_cat_brute = cont_cat_brute.style.format({"Bin Plot": fun, "Residual Plot": fun})


with open(final_report_link, "w") as f:
    f.write(
        "--------Correlation Matricies Table-------\n"
        + df_correlation_matricies.to_html()
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
        "\n-------Continous/Continous Brute Table-------\n" + cont_cont_brute.to_html()
    )
    f.write("\n\n")
    f.write(
        "-------Categorical/Categorical Correlation Table-------\n"
        + cat_cat_correlation.to_html()
    )
    f.write("\n\n")
    f.write(
        "-------Categorical/Categorical Brute Table-------\n" + cat_cat_brute.to_html()
    )
    f.write("\n\n")
    f.write(
        "-------Continous/Categorical Correlation Table-------\n"
        + cont_cat_correlation.to_html()
    )
    f.write("\n\n")
    f.write(
        "-------Continous/Categorical Brute Table-------\n" + cont_cat_brute.to_html()
    )
    f.write("\n\n")

    f.close()

webbrowser.open(urlpath)
