import json
import numpy as np
import cPickle as pickle
import redis

dimension = 100

with open("ids.json") as f:
    ids = json.load(f)

with open("titles.json") as f:
    titles = json.load(f)

r = redis.StrictRedis(host='localhost', port=6379, db=0)

for id,t in zip(ids, titles):
    r.set(id, t)
