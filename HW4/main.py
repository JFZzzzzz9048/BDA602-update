# import sys

# import matplotlib.pyplot as plt
# import numpy
import pandas as pd

# from plotly import figure_factory as ff
# import seaborn as sns
from plotly import express as px
from plotly import graph_objects as go
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix

# load dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)
# dataset only contains predictors
df = titanic["data"]

# add response variable
df["survived"] = titanic["target"]

# dataframe contains both a response and predictors
df = pd.DataFrame(df)

# Given a list of predictors and the response columns
response = ["survived"]
predictors = list(df.columns)
predictors.remove("survived")


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


print("Response '{}' is {}.".format(response[0], response_con_bool(df[response[0]])))


for i in predictors:
    print("Predictor '{}' is {}.".format(i, cont_bool(df[i])))


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


cat_response_cat_predictor(df["survived"], df["sex"])


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


cat_response_cont_predictor(df["survived"], df["age"])


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


cont_response_cat_predictor(df["age"], df["survived"])


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


cont_response_cont_predictor(df["age"], df["body"])
