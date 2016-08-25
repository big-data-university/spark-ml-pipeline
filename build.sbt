name := """spark-ml-pipeline"""

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0"
libraryDependencies += "com.esotericsoftware" %% "kryo" % "4.0.0"

