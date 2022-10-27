import os as os
from itertools import combinations, product

import data_process
import dataset_loader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plots
import seaborn as sns
from cat_correlation import (
    cat_cont_correlation_ratio,
    cat_correlation,
    con_con_correlation,
)
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


def diff_mean_response_2d(pred1, pred2, response, binNum):
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
    mean_square_difference = np.nanmean(mean_sqr_diff_weight)

    return mean_square_difference, weight_mean_square_difference


"""
def main():
    df, predictors, response = dataset_loader.get_test_data_set("titanic")
    print(df)
    print(predictors)
    print(response)


if __name__ == "__main__":
    sys.exit(main())
"""

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

    diff_mean_value, weighted_diff_mean_value = diff_mean_response_2d(
        df[cont_cont[0]], df[cont_cont[1]], df[response], 10
    )
    diff_mean.append(diff_mean_value)
    weighted_diff_mean.append(weighted_diff_mean_value)

# Correlation Table:
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

print(cont_cont_correlation)

fig = plt.figure(figsize=(15, 10))
sns.heatmap(df[cont_pred].corr(method="pearson"), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
# fig.show()

# Brute Force Table:
cont_cont_brute = pd.DataFrame(
    {
        "Predictor 1": predictor1,
        "Predictor 2": predictor2,
        "Difference of Mean Response": diff_mean,
        "Weighted Difference of Mean Response": weighted_diff_mean,
    }
)
print(cont_cont_brute)


# Categorical & Categorical
cat_predictors_column = []
cramers_v = []
cat_cat_absoulte = []
cat_cat_heatmap_link = []
cat_predictor1 = []
cat_predictor2 = []
cat_cat_diff_mean = []
cat_cat_weighted_diff_mean = []
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

    cat_cat_diff_mean_value, cat_cat_weighted_diff_mean_value = diff_mean_response_2d(
        labelencoder.fit_transform(df[cat_cat[0]]),
        labelencoder.fit_transform(df[cat_cat[1]]),
        df[response],
        10,
    )
    cat_cat_diff_mean.append(cat_cat_diff_mean_value)
    cat_cat_weighted_diff_mean.append(cat_cat_weighted_diff_mean_value)


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
print(cat_cat_correlation)


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
print(cat_cat_final_matrix)

# Correlation Plot
fig = plt.figure(figsize=(15, 10))
sns.heatmap(cat_cat_final_matrix, annot=True, cmap="Blues")
plt.title("Categorical-Categorical Correlation Heatmap")
# fig.show()


# Brute Force Table:
cat_cat_brute = pd.DataFrame(
    {
        "Predictor 1": cat_predictor1,
        "Predictor 2": cat_predictor2,
        "Difference of Mean Response": cat_cat_diff_mean,
        "Weighted Difference of Mean Response": cat_cat_weighted_diff_mean,
    }
)
print(cat_cat_brute)


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
        ) = diff_mean_response_2d(
            df[cont_predictor],
            labelencoder.fit_transform(df[cat_predictor]),
            df[response],
            10,
        )
        cont_cat_diff_mean.append(cont_cat_diff_mean_value)
        cont_cat_weighted_diff_mean.append(cont_cat_weighted_diff_mean_value)


# Correlation Table:
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
print(cont_cat_correlation)


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

print(cont_cat_matrix)


# Correlation Plot
fig = plt.figure(figsize=(15, 10))
sns.heatmap(cont_cat_matrix, annot=True, cmap="Blues")
plt.title("Continous-Categorical Correlation Heatmap")
# fig.show()

# Brute Force Table:
cont_cat_brute = pd.DataFrame(
    {
        "Predictor 1": cont_predictor1,
        "Predictor 2": cat_predictor2,
        "Difference of Mean Response": cont_cat_diff_mean,
        "Weighted Difference of Mean Response": cont_cat_weighted_diff_mean,
    }
)
print(cont_cat_brute)
