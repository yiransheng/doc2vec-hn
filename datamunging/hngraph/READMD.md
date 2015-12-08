# HN Graph

This Spark App reads HN Dataset from hdfs, and builds a graph with `child->parent` relations as edges. Next, it recursively finds ancestors for each node (comments) in the story using Spark Graphx Pregel API (max 1000 iterations, which should cover even the deepest comment tree). 

## Input

`/hndata/parquet_typed`

A single table with relavant fields: `id`, `type`, `parent`. 

## Output

Text file in the following format:

```
ID <TAB> TOP_LEVEL_ANCESTOR_ID,...,GRAND_PARENT_ID,PARENT_ID
```


