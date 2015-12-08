## Overview

The goal of this excercise is to train [Paragraph Vectors](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjc77TGksvJAhWPrIMKHTXQDjYQFggdMAA&url=https%3A%2F%2Fcs.stanford.edu%2F~quocle%2Fparagraph_vector.pdf&usg=AFQjCNE-2SsR07iQm78dQqVV-6_Y3ERurA&sig2=gVyET7lCT3Yx2-aPAJd4Lg) on Hacker News dataset (https://github.com/yiransheng/w251-project/blob/master/Google%20BigQuery), and uses resulting vectors to predict submission scores. 

## Process and Results

* 100-dimension vectors for all submission titles are trained using `gensim` and `Spark`.
* Random Forrest are used to predict submission scores, on dataset with 1810107 rows split into 0.7,0.3 training, test set
* MSE on the model is `1.1030` on `log(score)`, and `1745.7835` on raw scores

(The predictive power of the model is very _bad_, as the scores of submissions largely depends on submission contents rather than just titles, the same approach might have worked better on more "click-baity" titles)


## Machine Learning Pipeline

Two algorithums are performed on raw data:

1. Document Vector / Word Vector
2. Random Forrest

`Spark MLLib` has support for Random Forrest as well as `word2vec`, which runs efficiently on top of `RDD` abstraction. 

However, for document vector, is currently only implemented in python library `gensim` - and does not run in a distributed fashion. Normally the only way to use it would be exprot data from HDFS and run on single node. 


### Document Vector Using DeepDist 

[DeepDist](http://deepdist.com/) is a small python library to facilitate training deep networks in a distributed fashion on top of Spark.

> DeepDist implements Downpour-like stochastic gradient descent. It first starts a master model server. Subsequently, on each data node, DeepDist fetches the model from the master and calls gradient(). After computing gradients on the data partitions, gradient updates are sent back the the server. On the server, the master model is updated by descent(). Models can converge faster since gradient updates are constatently synchronized between the nodes.

### DeepDist Architecture

![](http://deepdist.com/images/deepdistdesign.png)

(Source:http://deepdist.com)


To install all dependencies and libraries across all nodes:
```
yum instal -y lapack, atlas, blas, lapack-devel, atlas-devel, blas-devel, numpy, scipy
pip install --upgrade setuptools
pip install gensim, deepdist, flask
```

Notice this setup is still far from perfect, and could not scale to larger dataset (for instance, full reddit dataset). The bottlenect is `gensim` requires raw data to be loaded into memory. `DeepDist` will distribute gradient decent across worker nodes, but it still needs to send the entire model datastructure down the wire to all nodes, and reconstruct `RDD`s locally:

```
with DeepDist(Word2Vec(corpus.collect()), master="spark://" + hosts["prj01"]) as dd:
    ...
``` 

Therefore, while this approach is helpful in speeding up training, it cannot scale beyond the limit of single node memory. Fortunately, Deep Learning as part of `MLLib` is being developped actively, and we may soon see better native support from `Spark`. 

### Random Forrest 


## Preprocessing

### Raw Data

Hackernews data dump till Oct. 2015, exported from: [https://bigquery.cloud.google.com/table/fh-bigquery:hackernews.full_201510](Google BigQuery). Uncompressed size: 4.3GB. It contains 20 million comments, and 2 million submissions. 

### Import/Export 

Python script `/datamunging/export-hn-submissions.py` is used to export data in csv format from BigQuery to Google Cloud Storage. Data is automatically split into 57 files. 

Next, I set permission to all 57 csv files to public, and use a simple shell script for a map-side only job on hadoop cluster to load data into HDFS, removing header rows at the same time (at location `/hndata/raw`). 

### Munging

Parsing csv is not simple task when one column contains `\n` (newline character), as each row could span across multiple lines. Thankfully, a row will not span across file boundrays due by BigQuery. 

One strategy to deal with this is csv SerDe for Hive. However, the overhead of parsing on every query is sub-optimal. In this case, I opted using `spark-csv` library to parse raw inputs, and write to HDFS in `parquet` format which is more efficient than csv and retains type information. 

__Step 1:__ Untyped Parquet 

Since all csv files are header-less, I have to define data schema in processing script, this is where I found out some limitations of `spark-csv`. For instance, we'd like to treat empty string as `NULL` (example is comments' title field, in raw csv it's recored as empty, unquoted strings), but `spark-csv` is not capable of handle it gracefully (see [this](https://github.com/databricks/spark-csv/issues/86) github issue).

As a compromise, I defined a simple schema with every column set to `StringType` and `Nullable`, and perform type conversion and handling of `null`s in the next step. 

Intermediate results of untyped data is written to `/hndata/parquet_untyped`

__Step 2:__ Typed Parquet 

In this step, I created a few sql `udf`s to transform columns. Most of these utilizes `scala`'s `Option` type. For example, the following UDF converts `StringType` column `deleted` to Nullable Boolean column:
```
sqlContext.udf.register("optBool", (s: String) => {
    Option(s).filter(_.trim.nonEmpty).map(_.toLowerCase).flatMap({
        case "true" => Some(true)
        case "false" => Some(false)
        case _ => None})
})
```

It's used like this:
```
"SELECT optBool(deleted) as deleted, ... ,  from hn"
```

Final output is stored at: `/hndata/parquet_typed`

