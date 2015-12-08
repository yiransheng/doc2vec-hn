# HN Typed

This Scala App reads untyped HN dataset, and convert columns to proper type and handles correctly NULL values. 

## Input
```
/hndata/parquet_untyped
```

Input columns all have `StringType`, and not NULL (instead NULL is stored as empty String).

## Output

Output is the same table wit same structure/columns and the following types:

```
Field       Type            Nullable

author      StringType      true
score       IntegerType     true
timestamp   TimestampType   true
title       StringType      true
type        StringType      true
url         StringType      true
text        StringType      true
parent      LongType        true
deleted     BooleanType     true
dead        BooleanType     true
descendants IntegerType     true
id          LongType        false
ranking     IntegerType     true
```


