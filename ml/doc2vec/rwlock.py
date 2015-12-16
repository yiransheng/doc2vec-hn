from threading import Thread
import urllib2
import urlparse
import threading

"""Simple reader-writer locks in Python
Many readers can hold the lock XOR one and only one writer

http://majid.info/blog/a-reader-writer-lock-for-python/
"""

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







