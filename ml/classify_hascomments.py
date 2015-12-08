from math import log, exp
from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()
sqlContext = SQLContext(sc)


df = sqlContext.read.load("hdfs:///hndata/parquet_typed", format="parquet")

stories = df.where("score IS NOT NULL") \
         .where("type='story'") \
         .where("title IS NOT NULL") 

commented = stories.map(lambda row: row.descendants) \
              .zipWithIndex() \
              .filter(lambda (d, _): d > 0).map(lambda (d, i): (True, i))

uncommented = stories.map(lambda row: row.descendants) \
              .zipWithIndex() \
              .filter(lambda (d, _): not d).map(lambda (d, i): (False, i))

ncommented = commented.count()
nuncommented = uncommented.count()

nsample = float( min(ncommented, nuncommented) )


data = commented.sample(False, nsample/ncommented) + uncommented.sample(False, nsample/nuncommented)
data.cache()

def loadVecs(score_pairs):
    import numpy as np
    docvecs = np.load("/data/_hndata/doc2vec_model/hn.docvecs.doctag_syn0.npy")
    return [(s, docvecs[i]) for (s,i) in score_pairs]

vecs = data.mapPartitions(loadVecs)
data = vecs.map(lambda pair: LabeledPoint(1.0 if pair[0] else 0.0, pair[1]))

# MLUtils.saveAsLibSVMFile(data, "hdfs:///hndata/spam_docvecs")

# Split the data into training and test sets
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.

rr = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=3, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)

predictions = rr.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
posErr = float( labelsAndPredictions.filter(lambda (v, p): v == 0.0 and v != p).count() ) / testData.filter(lambda lp: lp.label == 0.0).count()
negErr = float( labelsAndPredictions.filter(lambda (v, p): v == 1.0 and v != p).count() ) / testData.filter(lambda lp: lp.label == 1.0).count()
print('False Positive = ' + str(posErr))
print('False Negative = ' + str(negErr))
print('Learned classification forest model:')
print(rr.toDebugString())

rr.save(sc, "hdfs:///hndata/rrclassmodel_hascomment")
# sameModel = RandomForestModel.load(sc, "hdfs:///hndata/hnclassificationspam")
