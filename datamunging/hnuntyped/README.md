# HN Untyped

This Spark App reads csv files from HDFS and save to HDFS parquet. 

## Input

```
/hndata/raw
```

Input files are 57 header-less csvs. `comment` field do contain newline characters(`\n`). Therefore, we cannot process it line by line. `spark-csv` library is used for parsing with a custom schema:

```
  val hnSchema = StructType(Seq(
    StructField("author", StringType, true),
    StructField("score", StringType, true),
    StructField("time", StringType, true),
    StructField("title", StringType, true),
    StructField("type", StringType, true),
    StructField("url", StringType, true),
    StructField("text", StringType, true),
    StructField("parent", StringType, true),
    StructField("deleted", StringType, true),
    StructField("dead", StringType, true),
    StructField("descendants", StringType, true),
    StructField("id", StringType, false),
    StructField("ranking", StringType, true)))

```

We choose to convert `StringType` to proper data types in the next stage, as `spark-csv` has issues in inferring `NULL` values from empty String as well as converting some rows to proper types if we choose to define a more-restrictive schema. 


## Output

Output files are `parquet` files stored at `/hndata/parquet_untyped`
