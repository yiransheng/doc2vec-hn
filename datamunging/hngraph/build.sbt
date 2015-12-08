lazy val common = Seq(
  organization := "mids",
  version := "0.1.0",
  scalaVersion := "2.10.4",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "1.5.2" % "provided",
    "org.apache.spark" %% "spark-graphx" % "1.5.2" % "provided",
    "org.apache.spark" %% "spark-sql" % "1.5.2" % "provided"
  ),
  mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
     {
      case PathList("META-INF", xs @ _*) => MergeStrategy.discard
      case x => MergeStrategy.first
     }
  }
)

lazy val hn = (project in file(".")).
  settings(common: _*).
  settings(
    name := "hnGraph",
    mainClass in (Compile, run) := Some("HNGraph"))
