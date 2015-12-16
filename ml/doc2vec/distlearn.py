from threading import Thread
import urllib2
import urlparse
import threading

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
        self.bmodel = None
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
            bmodel = None
            self.lock.acquire_read()
            if not self.bmodel:
                self.lock.release()
                self.lock.acquire_write()
                if not self.bmodel:
                    self.bmodel = self.sc.broadcast(self.model)
                self.served += 1
                bmodel = self.bmodel
                self.lock.release()
            else:
                self.served += 1
                bmodel = self.bmodel
                self.lock.release()
            return "OK"


        @app.route('/update', methods=['GET', 'POST', 'PUT'])
        def update_flask():
            self.lock.acquire_write()
            if self.min_updates <= self.served:
                self.state = 'receiving'
            self.received += 1

            self.descent(self.model, gradient)

            if self.received >= self.served and self.min_updates <= self.received:
                self.received = 0
                self.served   = 0
                self.state    = 'serving'
                if self.bmodel is not None:
                    self.bmodel.unpersist(blocking=True)
                    self.bmodel = self.sc.broadcast(self.model)

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
            mod = self.bmodel.value
            grad = gradient(mod, data)
            self.amodel.add(grad)
            return [send_gradient(grad, master=master)]

        return rdd.mapPartitions(mapPartitions).collect()


def send_gradient(gradient, master='localhost:5000'):
    if not gradient:
        return 'EMPTY'
    request = urllib2.Request('http://%s/update' % master, pickleDumper.dumps(gradient, -1),
        headers={'Content-Type': 'application/deepdist'})
    return urllib2.urlopen(request).read()







