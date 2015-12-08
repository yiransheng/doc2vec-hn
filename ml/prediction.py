from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

from gensim.models.doc2vec import Doc2Vec
from math import exp
from threading import Thread, Event

sc = SparkContext()
sqlContext = SQLContext(sc)

# this is a large object we cache it on each worker node
gmod_broadcast = sc.broadcast( Doc2Vec.load("/data/_hndata/doc2vec_model/hn") ) 

tfidf_model = RandomForestModel.load(sc, "hdfs:///hndata/hnrrmodel_tfidf")

doc2vec_model = RandomForestModel.load(sc, "hdfs:///hndata/rrscoremodel")
doc2vec_model2 = RandomForestModel.load(sc, "hdfs:///hndata/rrscoremodel2")

tf = sc.pickleFile("hdfs:///hndata/tf_pickle")
idf = IDF().fit(tf)
hashingTF = HashingTF(1000)



def pred_tfidf(docs):
    sents = sc.parallelize(docs).map(lambda d: d.strip().split())
    new_tf = hashingTF.transform(sents)
    tfidf = idf.transform(new_tf)
    return tfidf_model.predict(tfidf)

def pred_doc2vec(docs, takelog=True, cased=False):
    sents = sc.parallelize(docs) \
              .map(lambda d: (d.lower() if not cased else d).strip().split())
    def loadDoc2vec(sents):
         gmod = gmod_broadcast.value 
         return [gmod.infer_vector(s) for s in sents]
    docvecs = sents.mapPartitions(loadDoc2vec)
    if takelog:
        preds = doc2vec_model.predict(docvecs)
        return preds.map(lambda x: exp(x) - 1.0).zip(docvecs.map(lambda x: x.tolist()))
    else:
        return doc2vec_model2.predict(docvecs).zip(docvecs.map(lambda x: x.tolist())) 

    
def serve():
    import sys
    import json
    from flask import Flask, request, jsonify, abort

    app = Flask(__name__)
    @app.route('/')
    def index():
        return 'Hello.'

    @app.route('/predict', methods=['GET'])
    def predict_submission():
        try:
           data = json.loads(request.args.get("data"))
           sentences = data["sentences"] 
        except Exception:
           abort(400)

        if len(sentences) > 0:
            preds = {}
            preds["tfidf"] = pred_tfidf(sentences).collect()
            preds["docvec"] = pred_doc2vec(sentences).collect()
            preds["docvec2"] = pred_doc2vec(sentences, False).map(lambda (p, v): p).collect()
            preds["sentences"] = sentences
            return jsonify(preds)
        else:
            preds = {}
            preds["tfidf"] = [] 
            preds["docvec"] = []
            preds["docvec2"] = []
            preds["sentences"] = sentences
            return jsonify(preds)

    def shutdown_server():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        shutdown_server()
        return 'Server shutting down...'
       
    try:
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        shutdown_server()
        sys.exit()

# by default Spark broadcast is lazy, we run predictions mannually to force 
# worker to cache gensim model 
pred_doc2vec(["Hello world", "Broadcast data", "Cache on worker"]).collect()


try:
    t = Thread(target=serve)
    t.start()
    t.join()
except (KeyboardInterrupt, SystemExit):
    sys.exit()
    
         
    

