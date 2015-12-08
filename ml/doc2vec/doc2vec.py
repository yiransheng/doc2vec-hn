from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from pyspark import SparkContext
from pyspark.sql import SQLContext

import os, shutil
import numpy as np
import copy
import cPickle as pickle
import cloudpickle as pickleDumper
from multiprocessing import Process
import socket
import sys
import os
from threading import Thread
import urllib2
import urlparse
import itertools, time

hosts = dict()
with open("/etc/hosts", 'r') as f:
    lines = f.readlines()
    for line in lines:
        ip, host = line.strip().split()[:2]
        print ("%s -> %s" % (host, ip))
        hosts[host] = ip


sc = SparkContext()
sqlContext = SQLContext(sc)

corpus_path = "hdfs:///hnio/submissions"
corpus_exists = os.system("hadoop fs -test -d %s" % corpus_path.replace("hdfs://", "")) == 0

if corpus_exists:
    corpus = sc.textFile(corpus_path)
    def mapLine(line):
        sentence, id = line.split("\t")
        return LabeledSentence(words=sentence.lower().split(), tags=[id])
    corpus = corpus.map(mapLine)

else:
    df = sqlContext.read.load("hdfs:///hndata/parquet_typed", format="parquet")
    submissionsDf = df.where("type='story'") \
                      .where("title IS NOT NULL") \
                      .where("id IS NOT NULL") \
                      .select("title", "id")

    def toLine(row):
        sentence = row.title.replace("\n", " ")
        sentence = sentence.replace("\t", " ")
        id = "TITLE_%d" % row.id
        return sentence + "\t" + id

    submissionsDf.map(toLine).saveAsTextFile(corpus_path)
    corpus = submissionsDf.map(lambda x: LabeledSentence(words=x.title.split(), tags=["TITLE_%d" % x.id]))


"""Simple reader-writer locks in Python
Many readers can hold the lock XOR one and only one writer

http://majid.info/blog/a-reader-writer-lock-for-python/
"""
import threading

version = """$Id: 04-1.html,v 1.3 2006/12/05 17:45:12 majid Exp $"""

class RWLock:
  """
A simple reader-writer lock Several readers can hold the lock
simultaneously, XOR one writer. Write locks have priority over reads to
prevent write starvation.
  """
  def __init__(self):
    self.rwlock = 0
    self.writers_waiting = 0
    self.monitor = threading.Lock()
    self.readers_ok = threading.Condition(self.monitor)
    self.writers_ok = threading.Condition(self.monitor)

  def acquire_read(self):
    """Acquire a read lock. Several threads can hold this typeof lock.
It is exclusive with write locks."""
    self.monitor.acquire()
    while self.rwlock < 0 or self.writers_waiting:
      self.readers_ok.wait()
    self.rwlock += 1
    self.monitor.release()

  def acquire_write(self):
    """Acquire a write lock. Only one thread can hold this lock, and
only when no read locks are also held."""
    self.monitor.acquire()
    while self.rwlock != 0:
      self.writers_waiting += 1
      self.writers_ok.wait()
      self.writers_waiting -= 1
    self.rwlock = -1
    self.monitor.release()

  def release(self):
    """Release a lock, whether read or write."""
    self.monitor.acquire()
    if self.rwlock < 0:
      self.rwlock = 0
    else:
      self.rwlock -= 1
    wake_writers = self.writers_waiting and self.rwlock == 0
    wake_readers = self.writers_waiting == 0
    self.monitor.release()
    if wake_writers:
      self.writers_ok.acquire()
      self.writers_ok.notify()
      self.writers_ok.release()
    elif wake_readers:
      self.readers_ok.acquire()
      self.readers_ok.notifyAll()
      self.readers_ok.release()

class DeepDist:
    def __init__(self, model, master='127.0.0.1:5000', min_updates=0, max_updates=4096):
        """DeepDist - Distributed deep learning.
        :param model: provide a model that can be trained in parallel on the workers
        """
        self.model  = model
        self.docvec_mapfile = model.docvecs.mapfile_path
        self.lock   = RWLock()
        self.descent  = lambda model, gradient: model
        self.master   = master
        self.state    = 'serving'
        self.served   = 0
        self.received = 0
        self.server   = '0.0.0.0'
        self.pmodel   = None
        self.np_docvecs = None
        self.min_updates = min_updates
        self.max_updates = max_updates

    def __enter__(self):
        Thread(target=self.start).start()
        # self.server = Process(target=self.start)
        # self.server.start()
        return self

    def __exit__(self, type, value, traceback):
        url = "http://%s:5000/shutdown"%self.server
        response = urllib2.urlopen(url, '{}').read()
        print"exit performed"

    def sync_cache(self, init=False):
        map_path = self.model.docvecs.mapfile_path
        self.model.docvecs.doctag_syn0.flush()
        # TO-DO: not hard-code this
        cmd1 = """for i in prj02 prj03 prj04 prj05; \
            do rsync -av %s.doctag_syn0 root@$i:%s.doctag_syn0; \
        done""" % (map_path, map_path)
        cmd2 = """for i in prj02 prj03 prj04 prj05; \
            do rsync -av %s.doctag_syn0_lockf root@$i:%s.doctag_syn0_lockf; \
        done""" % (map_path, map_path)
        if not init:
            return os.system(cmd1) == 0

        return os.system(cmd1) == 0 and os.system(cmd2) == 0

    def start(self):
        from flask import Flask, request

        app = Flask(__name__)

        @app.route('/')
        def index():
            return 'DeepDist'

        @app.route('/model', methods=['GET', 'POST', 'PUT'])
        def model_flask():
            i = 0
            while (self.state != 'serving' or self.served >= self.max_updates) and (i < 1000):
                time.sleep(1)
                i += 1

            # pickle on first read
            pmodel = None
            self.lock.acquire_read()
            if not self.pmodel:
                self.lock.release()
                self.lock.acquire_write()
                if not self.pmodel:
                    self.pmodel = pickle.dumps(self.model, -1)
                self.served += 1
                pmodel = self.pmodel
                self.lock.release()
            else:
                self.served += 1
                pmodel = self.pmodel
                self.lock.release()
            return pmodel

        @app.route('/docvecs', methods=['GET', 'POST', 'PUT'])
        def docvecs_flask():
            pass
            """
            i = 0
            while (self.state != 'serving' or self.served >= self.max_updates) and (i < 1000):
                time.sleep(1)
                i += 1

            np_docvecs = None
            self.lock.acquire_read()
            if not self.np_docvecs:
                self.lock.release()
                self.lock.acquire_write()
                if not self.np_docvecs:
                    map_path = self.model.docvecs.mapfile_path
                    doctag_syn0 = self.model.docvecs.doctag_syn0
                    doctag_syn0_lockf = self.model.docvecs.doctag_syn0_lockf
                    # length = max(len(self.model.docvecs.doctags),
                    #                 self.model.docvecs.count)
                    # doctag_syn0 = np.memmap(mapfile_path+'.doctag_syn0', dtype='float32',
                    #                      mode='r', shape=(length, self.model.vector_size))
                    # doctag_syn0_lockf = np.memmap(mapfile_path+'.doctag_syn0_lockf', dtype='float32',
                    #                      mode='r', shape=(length,))
                    self.np_docvecs = pickle.dumps((np.array(doctag_syn0),
                                                    np.array(doctag_syn0_lockf)), -1)
                self.served += 1
                np_docvecs = self.np_docvecs
                self.lock.release()
            else:
                self.served += 1
                np_docvecs = self.np_docvecs
                self.lock.release()
            return np_docvecs
            """

        @app.route('/update', methods=['GET', 'POST', 'PUT'])
        def update_flask():
            gradient = pickle.loads(request.data)

            self.lock.acquire_write()
            if self.min_updates <= self.served:
                state = 'receiving'
            self.received += 1

            self.descent(self.model, gradient)
            self.sync_cache()

            if self.received >= self.served and self.min_updates <= self.received:
                self.received = 0
                self.served   = 0
                self.state    = 'serving'
                self.pmodel = None
                self.np_docvecs = None

            self.lock.release()
            return 'OK'

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            return 'Server shutting down...'

        print 'Listening to 0.0.0.0:5000...'
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

    def train(self, rdd, gradient, descent):
        master = self.master   # will be pickled
        if master == None:
            master = rdd.ctx._conf.get('spark.master')
        if master.startswith('local['):
            master = 'localhost:5000'
        else:
            if master.startswith('spark://'):
                master = '%s:5000' % urlparse.urlparse(master).netloc.split(':')[0]
            else:
                master = '%s:5000' % master.split(':')[0]
        print '\n*** Master: %s\n' % master

        self.descent = descent

        self.sync_cache(True)
        def mapPartitions(data):
            mod = fetch_model(master=master)
            grad = gradient(mod, data)
            return [send_gradient(grad, master=master)]

        return rdd.mapPartitions(mapPartitions).collect()


def fetch_model(master=None):
    """
    req = urllib2.Request('http://%s/docvecs' % master,
        headers={'Content-Type': 'application/deepdist'})

    doctag_syn0, doctag_syn0_lockf = pickle.loads(urllib2.urlopen(req).read())
    doctag_syn0_mp = np.memmap(map_path + ".doctag_syn0", dtype=doctag_syn0.dtype, mode='w+',
                               shape = np.shape(doctag_syn0))
    doctag_syn0_lockf_mp = np.memmap(map_path + ".doctag_syn0_lockf",dtype=doctag_syn0_lockf.dtype,
                               mode='w+', shape = np.shape(doctag_syn0_lockf))
    doctag_syn0_mp[:] = doctag_syn0[:]
    doctag_syn0_lockf_mp[:] = doctag_syn0_lockf[:]
    doctag_syn0_mp.flush()
    doctag_syn0_lockf_mp.flush()
    """
    req = urllib2.Request('http://%s/model' % master,
        headers={'Content-Type': 'application/deepdist'})
    return pickle.loads(urllib2.urlopen(req).read())

def send_gradient(gradient, master='localhost:5000'):
    if not gradient:
        return 'EMPTY'
    request = urllib2.Request('http://%s/update' % master, pickleDumper.dumps(gradient, -1),
        headers={'Content-Type': 'application/deepdist'})
    return urllib2.urlopen(request).read()



model_path = "/root/doc2vec/doc2vec_model_final"
data_path = "/data/_hndata"

if not os.path.exists(model_path):
    os.mkdir(model_path)
else:
    for the_file in os.listdir(model_path):
        file_path = os.path.join(model_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception, e:
            print e


# init model
# corpus = corpus.sample(False, 0.0001)
data = corpus.collect()
model = Doc2Vec(size=100, max_vocab_size=long(4e5),
                docvecs_mapfile="%s/hndocvec_cache" % data_path)
model.build_vocab(data)
# adagrad params
model.ssyn0 = 0.0
model.ssyn1 = 0.0
model.docvecs.sdoctag_syn0 = 0.0
model.word_count = 0
model.total_words = len(model.vocab)
model.ver = 1
# merge with pre-trained word2vec model (from both comments and submissions)
# either use pretrained glove dataset (uncased) or word2vec trained from hn dataset
# model.intersect_word2vec_format("/root/doc2vec/word2vec_model/hn_compatible")
model.intersect_word2vec_format("/root/doc2vec/word2vec_model/glove_model.txt")


# Downpour adagrad implementation
def gradient(model, sentences): # executes on workers
    syn0,syn1,doctag_syn0 = model.syn0.copy(),model.syn1.copy(),model.docvecs.doctag_syn0.copy() # previous weights
    words = model.train(sentences, word_count=model.word_count,
                total_words=model.total_words)
    return {'syn0' : model.syn0 - syn0,
            'syn1' : model.syn1 - syn1,
            'words': words - model.word_count,
            'ver'  : model.ver,
            'doctag_syn0' : model.docvecs.doctag_syn0 - doctag_syn0}

def descent(model, update): # executes on master

    alpha = max(model.min_alpha,
                model.alpha * (1.0 - 1.0 * model.word_count / model.total_words))

    syn0 = update['syn0'] / alpha
    syn1 = update['syn1'] / alpha
    doctag_syn0 = update['doctag_syn0'] / alpha

    model.ssyn0 += syn0 * syn0
    model.ssyn1 += syn1 * syn1
    model.docvecs.sdoctag_syn0 += doctag_syn0 * doctag_syn0

    model.word_count = long(model.word_count) + long(update['words'])
    model.ver += 1

    alpha0 = alpha / (1e-6 + np.sqrt(model.ssyn0))
    alpha1 = alpha / (1e-6 + np.sqrt(model.ssyn1))
    alphad = alpha / (1e-6 + np.sqrt(model.docvecs.sdoctag_syn0))

    doctag_syn0_deltas = doctag_syn0 * alphad
    """
    global last_time
    if time.time() > last_time + 100:
        print ("**** docvecs ss deltas: %f ****" % (doctag_syn0_deltas * doctag_syn0_deltas).sum())
        last_time = time.time()
    """

    model.syn0 += syn0 * alpha0
    model.syn1 += syn1 * alpha1
    model.docvecs.doctag_syn0 += doctag_syn0_deltas

print "**** estimated base memory usage: ****"
print model.estimate_memory()


with DeepDist(model, master="spark://" + hosts["prj01"]) as dd:
    print("training start")
    # 300 iterations training
    for _ in itertools.repeat(None, 3):
        dd.train(corpus, gradient, descent)

    print("training done")
    dd.model.save("%s/hn" % model_path)



