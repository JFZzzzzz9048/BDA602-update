# import sys

from pyspark.sql import SparkSession

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
    SELECT * FROM baseball_ready;
    """

spark = SparkSession.builder.master("local[*]").getOrCreate()
df_batter_info = (
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

print(df_batter_info)
