from math import log, exp
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()
sqlContext = SQLContext(sc)


df = sqlContext.read.load("hdfs:///hndata/parquet_typed", format="parquet")

'''
def loadVecs(filename):
    npzfile = np.load(filename)
    return npzfile.iteritems()
vecs = vecFiles.flatMap(loadVecs)
'''

scores = df.where("score IS NOT NULL") \
         .where("type='story'") \
         .where("title IS NOT NULL") \
         .map(lambda row: row.score)

def loadVecs(score_pairs):
    import numpy as np
    docvecs = np.load("/data/_hndata/hn.docvecs.doctag_syn0.npy")
    return [(s, docvecs[i]) for (s,i) in score_pairs]

vecs = scores.zipWithIndex().mapPartitions(loadVecs)
data = vecs.map(lambda pair: LabeledPoint(log(float(pair[0])+1.0), pair[1]))

MLUtils.saveAsLibSVMFile(data, "hdfs:///hndata/docvecs")

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
labelsAndPredictions = testData.map(lambda lp: log(lp.label+1.0)).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest rr:')
print(rr.toDebugString())

rr.save(sc, "hdfs:///hndata/hnrrmodel")
# sameModel = RandomForestModel.load(sc, "hdfs:///hndata/hnrrmodel")
