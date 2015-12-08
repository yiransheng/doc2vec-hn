from math import log, exp
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()
sqlContext = SQLContext(sc)

# raw data
df = sqlContext.read.load("hdfs:///hndata/parquet_typed", format="parquet")

scores = df.where("score IS NOT NULL") \
         .where("type='story'") \
         .where("title IS NOT NULL") \
         .map(lambda row: row.descendants if row.descendants else 0)

def loadVecs(score_pairs):
    '''
    Executes on works, gensim doc2vec model has been rsynced to each 
    node on cluster, so each worker can read its own copy 

    As the model/np-array is larger than my driver memory, cannot use
    sc.broadcast to sync to each worker
    '''
    import numpy as np
    docvecs = np.load("/data/_hndata/doc2vec_model/hn.docvecs.doctag_syn0.npy")
    return [(s, docvecs[i]) for (s,i) in score_pairs]

# merge docvec2 with scores by index
vecs = scores.zipWithIndex().mapPartitions(loadVecs)
# map inputs to LabeledPoint for random forrest training
data = vecs.map(lambda pair: LabeledPoint(log(float(pair[0])+1.0), pair[1]))

# MLUtils.saveAsLibSVMFile(data, "hdfs:///hndata/docvecs_glove")

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

rr.save(sc, "hdfs:///hndata/rrcmtcountmodel")
