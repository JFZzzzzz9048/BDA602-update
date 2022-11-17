import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import graph_objects as go
from sklearn.metrics import confusion_matrix


# Build a function to plot Categorical Response by Categorical Predictor
def cat_response_cat_predictor(
    cat_response, cat_predictor, predictor_name, response_name
):
    file_location = "categorical_{}_categorical_{}_heatmap_plot.html".format(
        response_name, predictor_name
    )

    cat_predictor1, _ = pd.factorize(cat_predictor)
    cat_response1, _ = pd.factorize(cat_response)

    conf_matrix = confusion_matrix(cat_predictor1, cat_response1)
    """
    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    """

    fig_no_relationship = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=list(set(cat_predictor)),
            y=list(set(cat_response)),
            hoverongaps=False,
            zmin=0,
            zmax=conf_matrix.max(),
        )
    )

    fig_no_relationship.update_layout(
        title=f"Categorical Predictor {predictor_name} by Categorical Response {response_name} (with relationship)",
        xaxis_title=response_name,
        yaxis_title=predictor_name,
    )

    # fig_no_relationship.show()

    fig_no_relationship.write_html(
        file="plots/" + file_location,
        include_plotlyjs="cdn",
    )

    return predictor_name, file_location


# Build a function to plot Categorical Response by Continous Predictor
def cat_response_cont_predictor(
    cat_response, cont_predictor, predictor_name, response_name
):
    file_location_cat_cont_violin = (
        "continous_{}_categorical_{}_violin_plot.html".format(
            response_name, predictor_name
        )
    )

    file_location_cat_cont_dist = "continous_{}_categorical_{}_dist_plot.html".format(
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

    with open("plots/" + file_location_cat_cont_violin, "a") as f:
        f.write(violin.to_html(full_html=False, include_plotlyjs="cdn"))

    with open("plots/" + file_location_cat_cont_dist, "a") as f:
        f.write(hist.to_html(full_html=False, include_plotlyjs="cdn"))

    return predictor_name, file_location_cat_cont_violin, file_location_cat_cont_dist


# Build a function to plot Continous Response by Continous Predictor
def cont_response_cont_predictor(
    cont_response, cont_predictor, predictor_name, response_name
):
    file_location = "continous_{}_countinous_{}_plot.html".format(
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
    # scatter.show()

    scatter.write_html(
        file="plots/" + file_location,
        include_plotlyjs="cdn",
    )
    return predictor_name, file_location


# Perform a linear Regression and return p-value and t-score
def plot_linear(cont_response, cont_predictor, predictor_name, response_name):
    file_location = "{}_{}_linear_regression_plot.html".format(
        predictor_name, response_name
    )
    y = np.array(cont_response)
    predictor = np.array(cont_predictor)

    predictor1 = statsmodels.api.add_constant(np.array(predictor))
    linear_regression_model = statsmodels.api.OLS(y, predictor1)
    linear_regression_model_fitted = linear_regression_model.fit()

    # print(linear_regression_model_fitted.summary())

    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(x=predictor, y=y, trendline="ols")
    fig.update_layout(
        title=f"{predictor_name} vs. {response_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title="Variable: {}".format(predictor_name),
        yaxis_title=response_name,
    )
    # fig.show()

    fig.write_html(
        file="plots/" + file_location,
        include_plotlyjs="cdn",
    )
    return predictor_name, t_value, p_value, file_location
