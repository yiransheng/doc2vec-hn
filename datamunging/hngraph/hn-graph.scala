import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.graphx._

/**
 * HN Dataset Create Graph
 *
 * This program creates two textfiles on hdfs
 * 1. <comment id> TAB <top level parent id>
 * 2. <comment id> TAB <comma seperated list of full ancestry ids>
 * 
 * hdfs:///hndata/cmtparent_top
 * hdfs:///hndata/cmtancestry
 *
 * @author Yiran Sheng
 */


object HN extends App {
  // setting things up
  val conf = new SparkConf().setAppName("HNGraph")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  
  // full hn database
  val df = sqlContext.read.parquet("/hndata/parquet_typed")

  // subset to join
  var parents = df.select("id")
  val children = df.where("parent is NOT NULL")
                   .select("id", "parent")

  val joined = children.as("a")
     .join(parents.as("p"), $"a.parent" === $"p.id")
  val cmtedges = joined.select($"a.id", $"a.parent")
  
  // df -> string -> hdfs
  cmtedges.map(row => row(0).toString + "\t" + row(1).toString)
          .saveAsTextFile("hdfs:///hndata/cmtedges")
          
  val graph = GraphLoader.edgeListFile(sc, "hdfs:///hndata/cmtedges")
  // empty List for holding graph ancestory path for each node
  val nil:List[org.apache.spark.graphx.VertexId] = List()
  val initialGraph = graph.mapVertices((id, _) => nil)
  val sssp = initialGraph.pregel(0L, 1000)(   // max iteration is 1000 >> comment tree height
    // Vertex Program: 
    // ignore initial message (0L), otherwise append upstream ancestor id
    // to vertice attr list
    (id, attr, msg) => if(msg > 0L) (msg :: attr) else attr, 
    // Send Message:
    // triplet is: child(src) -> parent(dst)
    // if child and parent share common top level ancestor (head of attr list), stop
    // if parent has no parent (top level), and child's top level ancestor is parent, stop
    //    |
    //    |otherwise, if parent has no parent (top level), send parent id to child
    //    --------- 
    // if parent has top level parent, send its id to child
    triplet => {  
    
      if(triplet.dstAttr.nonEmpty && triplet.srcAttr.nonEmpty) {
        if(triplet.srcAttr.head == triplet.dstAttr.head) {
          Iterator.empty
        } else {
          Iterator((triplet.srcId, triplet.dstAttr.head))
        }
      } else {
        if(triplet.srcAttr.nonEmpty && triplet.srcAttr.head == triplet.dstId) {
          Iterator.empty
        } else { 
          Iterator((triplet.srcId, triplet.dstId))
        }  
      }
   
    },
    // input graph is a tree, each child recieves one msg from immediate parent
    // in each iteration, merge message function just takes it
    (a,b) => a // Merge Message
  )
  
  // save top level parent (submissions) to textFile on hdfs
  sssp.vertices
    .map{case (id, parentList) => id.toString + "\t" + (if(parentList.nonEmpty) parentList.head.toString else "")}
    .saveAsTextFile("hdfs:///hndata/cmtparent_top")
    
  // save ancestry tree (seperated by comma) to textFile on hdfs
  sssp.vertices
    .map{case (id, parentList) => id.toString + "\t" + parentList.map(_.toString).mkString(",")}
    .saveAsTextFile("hdfs:///hndata/cmtancestry")
    
}
