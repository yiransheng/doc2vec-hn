# Machine Learning

Files listed in this directory run the training of various models, to submit individual model:

```
$SPARK_HOME/bin/spark-submit --verbose \
   --master yarn \
   --deploy-mode client \
   --driver-java-options '-Xms3g -Xmx6g' \
   --driver-memory 6g \
   --executor-memory 6g \
   ./score_model.py
```

Currently the code is a bit dirty, using a lot of hardcoded file/hdfs paths in source code.

## Models

* `doc2vec/word2vec.py` Word2Vec model on all stories and comments
* `doc2vec/doc2vec.py` Doc2Vec model on all submissions, using either pretrained `glove.6B.100d` word vectors or output of the previous model
* `score_model.py` random forrerst, using docvecs trained in the above model to predict submission scores (taking the log of scores)
* `score_model2.py` same model on raw scores, without taking log
* `classify_spam.py` using docvecs to classify if a submission is "dead" or not
* `classify_hascomment.py` classify if a submission has comment or not
* `preditions.py` load trained models and run a api server to predict new submission title scores

