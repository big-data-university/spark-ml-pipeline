package com.bdu

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}



/**
 * Created by eric on 8/11/16.
 * Spark is already pretty fast, but can we push the boundary and make Spark 10X faster?
 * Spark 2.0 ships with the second generation Tungsten engine. This engine builds
 * upon ideas from modern compilers and MPP
 * databases and applies them to Spark workloads. The main idea is to emit
 * optimized code at runtime that collapses the entire
 * query into a single function, eliminating virtual function calls and leveraging
 * CPU registers for intermediate data. We
 * all this technique “whole-stage code generation.”
 *
 * spark.sql.codegen.wholeStage = true
 */
object Benchmark {

  // Define a simple benchmark util function


  def benchmark(name: String)(f: => Unit) {
    val startTime = System.nanoTime
    f
    val endTime = System.nanoTime
    println(s"Time taken in $name: " + (endTime - startTime).toDouble / 1000000000 + " seconds")
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Benchmark").setMaster("local[*]") //Using 2 core running on local
    val sc = new SparkContext(conf)

    val spark = SQLContext.getOrCreate(sc)
    sc.setLogLevel("ERROR")
    //val spark = new SQLContext(sc)

    //import spark.implicits._

    println(s"spark version : " + sc.version)



    spark.setConf("spark.sql.codegen.wholeStage", "false")

    benchmark("Spark 1.6 sum(id)") {
      spark.range(1000L * 1000 * 1000).show(5)
      spark.range(1000L * 1000 * 1000).selectExpr("sum(id)").show()
    }

    benchmark("Spark 1.6 join()") {
      val count = spark.range(1000L * 1000 * 1005).join(spark.range(1040L).toDF(), "id").count()
      println(count)
    }

    spark.setConf("spark.sql.codegen.wholeStage", "true")

    benchmark("Spark 2.0 sum(id)") {
      spark.range(1000L * 1000 * 1000).selectExpr("sum(id)").show()
      print(spark.range(1000L * 1000 * 1000).stat)
    }

    benchmark("Spark 2.0 join()") {
      val count = spark.range(1000L * 1000 * 1005).join(spark.range(1040L).toDF(), "id").count()
      println(count)
    }
  }

}
