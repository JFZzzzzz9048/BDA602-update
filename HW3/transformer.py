from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as func
from pyspark.sql.functions import col
from pyspark.sql.window import Window


class RollingAVGTransform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(RollingAVGTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        w = (
            Window()
            .partitionBy(dataset[input_cols[0]])
            .orderBy(dataset[input_cols[1]].cast("timestamp").cast("long"))
            .rangeBetween(-8640000, -1)
        )

        dataset = dataset.withColumn(
            "SUM_of_atBat", func.sum(dataset[input_cols[2]]).over(w)
        ).withColumn("SUM_of_Hit", func.sum(dataset[input_cols[3]]).over(w))
        dataset = dataset.withColumn(
            output_col, col("SUM_of_Hit") / col("SUM_of_atBat")
        )
        dataset = dataset.sort("Batter_ID", "Game_Date").na.drop()

        return dataset
