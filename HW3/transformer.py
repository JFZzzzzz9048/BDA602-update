from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.functions import col
from pyspark.sql.window import Window

database = "baseball"
username = "root"
pw = "password"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"
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


class RollingAVGTransform(Transformer):
    @keyword_only
    def __init__(self):
        super(RollingAVGTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self):
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

        w = (
            Window()
            .partitionBy(df_batter_info["Batter_ID"])
            .orderBy(df_batter_info["Game_Date"].cast("timestamp").cast("long"))
            .rangeBetween(-8640000, -1)
        )

        df = df_batter_info.withColumn("SUM_of_atBat", func.sum("atBat").over(w))
        df = df.withColumn("SUM_of_Hit", func.sum("Hit").over(w))
        df = df.withColumn("Rolling_Average", col("SUM_of_Hit") / col("SUM_of_atBat"))
        df = df.sort("Batter_ID", "Game_Date").na.drop()

        return df
