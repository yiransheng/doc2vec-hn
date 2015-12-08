from pyspark import SparkContext
from pyspark.sql import SQLContext

from gensim.models.doc2vec import Doc2Vec

sc = SparkContext()
sqlContext = SQLContext(sc)

# this is a large object we cache it on each worker node
gmod_broadcast = sc.broadcast( Doc2Vec.load("/root/doc2vec/doc2vec_model/hn") ) 

df = sqlContext.read.load("hdfs:///hndata/parquet_typed", format="parquet")

ids = df.where("score IS NOT NULL") \
         .where("type='story'") \
         .where("title IS NOT NULL") \
         .map(lambda row: row.id)

def mergeVec(id):
    gmod = gmod_broadcast.value 
    vec = gmod.docvecs["TITLE_%d" % id]
    return (id, vec)
    
docvecs = ids.map(mergeVec) 
docvecs.saveAsPickleFile("hdfs:///hndata/docvecs_glove_pickle")
