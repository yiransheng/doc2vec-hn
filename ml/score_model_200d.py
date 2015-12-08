from math import log, exp
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()
sqlContext = SQLContext(sc)

docvec1 = MLUtils.loadLibSVMFile(sc, "hdfs:///hndata/docvecs")
docvec2 = MLUtils.loadLibSVMFile(sc, "hdfs:///hndata/score_data_docvecs")

data = docvec1.zip(docvec2) \
         .map(lambda (lp1, lp2): LabeledPoint(lp1.label, np.concatenate([lp1.features, lp2.features])))

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

rr.save(sc, "hdfs:///hndata/rrscoremodel2")
