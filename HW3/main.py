import sys

from pyspark.sql import SparkSession
from transformer import RollingAVGTransform


def main():
    # Setup Spark
    database = "baseball"
    username = "root"
    pw = "password"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # SQL query to generate a batter info dataframe
    sql_batter_info = """
    SELECT
    BC.batter AS Batter_ID,
    SUM(BC.atBat) AS atBat,
    SUM(BC.Hit) AS Hit,
    G.game_ID AS game_ID,
    DATE(G.local_date) AS Game_Date
    FROM batter_counts BC
    JOIN game G
    ON G.game_id = BC.game_id
    GROUP BY BC.batter, DATE(G.local_date)
    """

    spark = SparkSession.builder.master("local[*]").getOrCreate()
    df_batter_info = (
        spark.read.format("jdbc")
        .options(
            url=jdbc_url,
            query=sql_batter_info,
            user=username,
            password=pw,
            driver=jdbc_driver,
        )
        .load()
    )

    # Generate a rolling average dataframe through transformer
    rolling = RollingAVGTransform(
        inputCols=["Batter_ID", "Game_Date", "atBat", "Hit"],
        outputCol="Rolling_Average",
    )
    rolling_avg = rolling._transform(df_batter_info)
    print("The first 30 rows of the 100 days rolling average table: ")
    rolling_avg.show(30)
    return


if __name__ == "__main__":
    sys.exit(main())
