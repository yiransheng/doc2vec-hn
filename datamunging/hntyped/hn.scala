import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, TimestampType, BooleanType};
import org.apache.spark.sql.functions.{lit, when, from_unixtime}

/**
 * HN Dataset csv -> Hive 
 * 
 * @author Yiran Sheng
 */
object HNSql extends App {
  // create SparkConf, SparkContext & SQLContext
  val conf = new SparkConf().setAppName("HN Typed")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)


  val parquetFile = sqlContext.read.parquet("/hndata/parquet_untyped")
  // register temp data hnUntyped
  // hnUntyped has all columns as StringType, and empty String is preserved as-is
  parquetFile.registerTempTable("hnUntyped")


  // udfs - convert untyped hn data to proper conlumn types
  // 
  // we treat empty string as null (by returning None in udfs)

  sqlContext.udf.register("emptyAsNull", 
    (s: String) => Option(s).filter(_.trim.nonEmpty))

  sqlContext.udf.register("optInt", (s: String) => {
    try {
      Some(s.trim.toInt)
    } catch {
      case e: Exception => None
    }    
  })
  sqlContext.udf.register("optLong", (s: String) => {
    try {
      Some(s.trim.toLong)
    } catch {
      case e: Exception => None
    }    
  })
  sqlContext.udf.register("optBool", (s: String) => {
    Option(s).filter(_.trim.nonEmpty).map(_.toLowerCase).flatMap({
        case "true" => Some(true)
        case "false" => Some(false)
        case _ => None})
  })

  val typed1 = sqlContext.sql("""SELECT 
    emptyAsNull(author) as author,
    optInt(score) as score,
    optLong(time) as timestamp,
    emptyAsNull(title) as title,
    emptyAsNull(type) as type,
    emptyAsNull(url) as url, 
    emptyAsNull(text) as text,
    optLong(parent) as parent,
    optBool(deleted) as deleted,
    optBool(dead) as dead,
    optInt(descendants) as descendants,
    optLong(id) as id,
    optInt(ranking) as ranking FROM hnUntyped""")

  val typed = typed1
    .withColumn("timestamp", from_unixtime(typed1("timestamp")))  // Long -> String/TimeStamp (hive compatibility)
    .filter("id IS NOT NULL")

  typed.write.parquet("hdfs:///hndata/parquet_typed")

}
