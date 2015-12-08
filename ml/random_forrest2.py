from math import log, exp
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()
sqlContext = SQLContext(sc)


data = MLUtils.loadLibSVMFile(sc, "hdfs:///hndata/docvecs")
data = data.map(lambda lp: LabeledPoint(exp(lp.label)-1.0, lp.features))

# Split the data into training and test sets
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
rr = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                    numTrees=5, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32)

predictions = rr.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest rr:')
print(rr.toDebugString())

rr.save(sc, "hdfs:///hndata/hnrrmodel2")
# sameModel = RandomForestModel.load(sc, "hdfs:///hndata/hnrrmodel")
