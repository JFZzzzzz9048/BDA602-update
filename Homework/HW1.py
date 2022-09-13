import numpy
import pandas
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    # Load the iris data set
    iris_df = pandas.read_csv(
        "https://teaching.mrsharky.com/data/iris.data", header=None
    )
    iris_df.head()
    iris_df.tail()

    # Attribute name dictionary
    attribute_dic = {
        "1": "sepal length in cm",
        "2": "sepal width in cm",
        "3": "petal length in cm",
        "4": "petal width in cm",
    }

    # Rename the columns
    iris_df.columns = [
        attribute_dic["1"],
        attribute_dic["2"],
        attribute_dic["3"],
        attribute_dic["4"],
        "Name",
    ]
    iris_df.head()

    # Check how many names in data set
    index_name = list(set(iris_df["Name"]))

    # Get some simple summary statistics (mean, min, max, quartiles) using numpy
    # Check dimensions
    iris_np = iris_df.to_numpy()
    # iris_np.shape

    # Simple statistics(min, max, mean, quantiles)
    min_np = list(iris_np[:, :4].min(axis=0))
    max_np = list(iris_np[:, :4].max(axis=0))

    mean_np = list(iris_np[:, :4].mean(axis=0))
    mean_np = [round(item, 2) for item in mean_np]

    Q1_np = list(numpy.quantile(iris_np[:, :4], 0.25, axis=0))
    Q2_np = list(numpy.quantile(iris_np[:, :4], 0.50, axis=0))
    Q3_np = list(numpy.quantile(iris_np[:, :4], 0.75, axis=0))

    stats_np = numpy.dstack((min_np, max_np, mean_np, Q1_np, Q2_np, Q3_np))
    stats_np = stats_np[0, :]

    final_stats_df = pandas.DataFrame(
        stats_np,
        columns=["Min", "Max", "Mean", "Q1", "Q2", "Q3"],
        index=[
            attribute_dic["1"],
            attribute_dic["2"],
            attribute_dic["3"],
            attribute_dic["4"],
        ],
    )
    print(final_stats_df)

    # Plot the different classes against one another (try 5 different plots)
    # 1. Scatter Plot
    fig = px.scatter(
        iris_df,
        x="sepal width in cm",
        y="sepal length in cm",
        color="Name",
        size="petal length in cm",
        hover_data=["petal width in cm"],
    )
    fig.show()

    # 2. Violin Plot
    fig = go.Figure()
    for name in index_name:
        fig.add_trace(
            go.Violin(
                x=iris_df["Name"][iris_df["Name"] == name],
                y=iris_df["sepal length in cm"][iris_df["Name"] == name],
                name=name,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.show()

    # 3. Boxplot(by name)
    fig = px.box(iris_df, x="Name", y="sepal length in cm", color="Name")
    fig.update_traces(
        quartilemethod="exclusive"
    )  # or "inclusive", or "linear" by default
    fig.show()

    # 4. Boxplot
    y0 = iris_df[attribute_dic["1"]]
    y1 = iris_df[attribute_dic["2"]]
    y2 = iris_df[attribute_dic["3"]]
    y3 = iris_df[attribute_dic["4"]]

    fig = go.Figure()
    fig.add_trace(go.Box(y=y0, name=attribute_dic["1"]))
    fig.add_trace(go.Box(y=y1, name=attribute_dic["2"]))
    fig.add_trace(go.Box(y=y2, name=attribute_dic["3"]))
    fig.add_trace(go.Box(y=y3, name=attribute_dic["4"]))

    fig.show()

    # 5. Line plot
    y0 = iris_df[attribute_dic["1"]]
    y1 = iris_df[attribute_dic["2"]]
    y2 = iris_df[attribute_dic["3"]]
    y3 = iris_df[attribute_dic["4"]]

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y0, mode="lines", name=attribute_dic["1"]))
    fig.add_trace(go.Scatter(y=y1, mode="lines", name=attribute_dic["2"]))
    fig.add_trace(go.Scatter(y=y2, mode="lines", name=attribute_dic["3"]))
    fig.add_trace(go.Scatter(y=y3, mode="lines", name=attribute_dic["4"]))
    fig.show()

    # Change Categorical variable to numerical (dummy code)
    ord_enc = OrdinalEncoder()
    iris_df[["Name"]] = ord_enc.fit_transform(iris_df[["Name"]])

    # DataFrame to numpy values
    x = iris_df[
        [attribute_dic["1"], attribute_dic["2"], attribute_dic["3"], attribute_dic["4"]]
    ]
    y = iris_df["Name"]

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=150, stratify=y
    )

    # Use StandardScaler() transform
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use RandomForestClassifier(not in pipeline)
    rfr = RandomForestClassifier().fit(X_train_scaled, y_train)
    r2 = rfr.score(X_test_scaled, y_test)
    print("Not in Pipeline the accuracy of Random Forest is: {}".format(r2))

    # Pipeline Random Forest
    pipe_random_forest = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=150)),
        ]
    )
    pipe_random_forest.fit(X_train, y_train)
    pipe_random_forest.score(X_test, y_test)
    print(
        "The accuracy of Random Forest is: {}".format(
            pipe_random_forest.score(X_test, y_test)
        )
    )

    # Pipeline Decision Tree
    pipe_decision_tree = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=150)),
        ]
    )
    pipe_decision_tree.fit(X_train, y_train)
    pipe_decision_tree.score(X_test, y_test)
    print(
        "The accuracy of Decision Tree is: {}".format(
            pipe_decision_tree.score(X_test, y_test)
        )
    )
    return


if __name__ == "__main__":
    exit(main())
