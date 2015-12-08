import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, TimestampType, BooleanType, LongType};
import org.apache.spark.sql.functions.from_unixtime

/**
 * HN Dataset csv -> Hive 
 * 
 * @author Yiran Sheng
 */
object HNSql extends App {
  // create SparkConf, SparkContext & SQLContext
  val conf = new SparkConf().setAppName("HN Sql")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)

  val csvPath = "hdfs:///hndata/raw/"
  // orig csv header: by,score,time,title,type,url,text,parent,deleted,dead,descendants,id,ranking
  // raw csv data files are headerless
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

  val df = sqlContext.read
    .format("com.databricks.spark.csv")
    .option("header", "false")   
    .schema(hnSchema)
    .load(csvPath)

  df.registerTempTable("hn")

  df.write.parquet("hdfs:///hndata/parquet_untyped")

}
