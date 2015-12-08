from gensim.models.word2vec import Word2Vec
from pyspark import SparkContext
from pyspark.sql import SQLContext

from cStringIO import StringIO
from HTMLParser import HTMLParser

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

hosts = dict()
with open("/etc/hosts", 'r') as f:
    '''
    Read /etc/hosts to map host name to ip address later
    '''
    lines = f.readlines()
    for line in lines:
        ip, host = line.strip().split()[:2]
        print ("%s -> %s" % (host, ip))
        hosts[host] = ip



# init spark
sc = SparkContext()
sqlContext = SQLContext(sc)

# raw data
df = sqlContext.read.load("hdfs:///hndata/parquet_typed", format="parquet")

# filter all submissions/stories
submissionsDf = df.where("type='story'") \
                  .where("title IS NOT NULL") \
                  .where("id IS NOT NULL") \
                  .select("title")

# filter all comments
commentsDf = df.where("type = 'comment'") \
               .where("text IS NOT NULL") \
               .select("text")

class HTMLTagStripper(HTMLParser):
    '''
    Extends HTMLParser to strip out html tags
  
    comments ofter contains tags such as <p>, <a>, <pre> <code>
    we strip all of them, and insert "\n" for <p> tags
    '''
    def __init__(self, data):
        self.plaintext = ""
        HTMLParser.__init__(self)
        self.feed(data)
    def handle_data(self, data):
        tag = self.get_starttag_text()
        if len(data) == 0:
            return 
        if tag == "<p>":
            self.plaintext = self.plaintext.strip() + ("\n" + data.strip() + "\n")
        else:
            self.plaintext += (" " + data.strip())
            
    def plain(self):
        return self.plaintext.strip()

# simple parser, just split by white space
submission_sentences = submissionsDf.map(lambda row: row.title.strip().split())

# strip tags
def parseComment(comment):
    try:
        comment = HTMLTagStripper(comment).plain()
    except Exception, e:
        comment = "" 
    if not comment or len(comment) == 0:
        return [[]]
    paragraphs = comment.split("\n")
    return [[w for w in s.split() if len(w) < 50] for s in paragraphs]

comments_sentences = commentsDf.flatMap(lambda row: parseComment(row.text))

# union two rdds
sentences = submission_sentences + comments_sentences

# calculate the deltas in model params (np-arrays)
def gradient(model, sentences): # executes on workers
    syn0, syn1 = model.syn0.copy(), model.syn1.copy() # previous weights
    model.train(sentences)
    return {'syn0': model.syn0 - syn0, 'syn1': model.syn1 - syn1}
 
# apply deltas computed on works 
def descent(model, update): # executes on master
    model.syn0 += update['syn0']
    model.syn1 += update['syn1']

# DeepDist: http://deepdist.com/
# DeepDist dependencies and src below, some tweaks are made

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
        self.lock   = RWLock()
        self.descent  = lambda model, gradient: model
        self.master   = master
        self.state    = 'serving'
        self.served   = 0
        self.received = 0
        self.server   = '0.0.0.0'
        self.pmodel   = None
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
    

        @app.route('/update', methods=['GET', 'POST', 'PUT'])
        def update_flask():
            gradient = pickle.loads(request.data)

            self.lock.acquire_write()
            if self.min_updates <= self.served:
                state = 'receiving'
            self.received += 1
            
            self.descent(self.model, gradient)
            
            if self.received >= self.served and self.min_updates <= self.received:
                self.received = 0
                self.served   = 0
                self.state    = 'serving'
                self.pmodel = None
            
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
        
        model = self.model
        def mapPartitions(data):
            mod = fetch_model(master=master)
            grad = gradient(mod, data)
            return [send_gradient(grad, master=master)]
        
        return rdd.mapPartitions(mapPartitions).collect()

def fetch_model(master='localhost:5000'):
    request = urllib2.Request('http://%s/model' % master,
        headers={'Content-Type': 'application/deepdist'})
    return pickle.loads(urllib2.urlopen(request).read())

def send_gradient(gradient, master='localhost:5000'):
    if not gradient:
        return 'EMPTY'
    request = urllib2.Request('http://%s/update' % master, pickleDumper.dumps(gradient, -1),
        headers={'Content-Type': 'application/deepdist'})
    return urllib2.urlopen(request).read()
 
import os, shutil 

model_path = "/root/doc2vec/word2vec_model"

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


model = Word2Vec(submission_sentences.collect() + comments_sentences.takeSample(False, 1000000, 1), 
                 size=100,
                 max_vocab_size=long(1e7))
print "**** estimated memory usage: ****"
print model.estimate_memory()
with DeepDist(model, master="spark://" + hosts["prj01"]) as dd:
    '''
    Distributed training
    '''
    print("training start")
    dd.train(sentences, gradient, descent)
    print("training done")
    dd.model.save("%s/hn" % model_path) 
    dd.model.save_word2vec_format("%s/hn_compatible" % model_path) 




   

