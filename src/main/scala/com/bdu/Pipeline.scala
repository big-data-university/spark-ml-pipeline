package com.bdu

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec}
import org.apache.spark.ml.tuning.{CrossValidatorModel, CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SizeEstimator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark._


import scala.io.Source


/**
 * Created by Eric on 8/11/16.
 *
 *  We built an pipleline for Spark ML to detect SpamSMS,
 *
 *  During this model training, we treat the Pipeline as an Estimator, wrapping it in
 *  CrossValidator instance. This will allow us to jointly choose parameters for all Pipeline stages.
 *  A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
 *  We use a ParamGridBuilder to construct a grid of parameters to search over.
 *  With 3 values for word2vec.vectorSize and 3 values for lr.regParam,
 *  this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from 9 x NumFolds
 *
 */

object Pipeline {

  final val LABEL_POSITIVE = 1.0
  final val LABEL_NEGATIVE = 0.0

  final val CLASS_SPAM = "spam"

  //final val TRAIN_DATA = "https://s3.amazonaws.com/workflowexecutor/examples/data/SMSSpamCollection.csv"

  /*
   *Using local data for quick test
   */
  final val TRAIN_DATA = "resource/SMSSpamCollection.csv"

  var model: CrossValidatorModel = null

  def main(args: Array[String]): Unit = {

    println("初始化SQLContext")
    val conf = new SparkConf().setAppName("CrossValidation Pipeline").setMaster("local[4]") //Using 2 core running on local



    /*
    *Enable Kryoserilizer for efficient ser
     */
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    /*
     *Increase parallelism, local default = number of core. e.g. 8 on mac
     */

    conf.set("spark.default.parallelism", "4")

    /**
     * Enable GC trace for executor
     */
    conf.set("spark.executor.extraJavaOptions", "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+UseCompressedOops")

    /**
     * Full GC happen for multiple times
     * JVM Parameters:
     * -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+UseCompressedOops
     */
    conf.set("spark.driver.memory", "2G") // Increase java heap for driver, -Xmx is illegal setting
    conf.set("spark.executor.memory", "2G")


    //conf.set("spark.executor.memory", "1G") //Increase to 2G per executor from default value 1G. -Xmx1G is illegal setting for spark executor.

    /**
     * Increase executor numbers to increase parallelism, only works for cluster env. standalone, yarn, mesos
     */
    //conf.set("spark.executor.instances", "4")

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = SQLContext.getOrCreate(sc)
    import sqlContext.implicits._
    sqlContext.setConf("spark.sql.codegen.wholeStage", "true")
    //sqlContext.setConf("spark.executor.instances", "8")




    /*
    * Split training set, 80% for training, 20% for test.
    * */

    val (spam, ham) = loadData(TRAIN_DATA)
    val (trainSpan, testSpan) = spam.partition(_._1.length % 10 < 8)
    val (trainHam, testHam) = ham.partition(_._1.length % 10 < 8)


    val negTest = sc.parallelize(testSpan).toDF("text", "label").persist(StorageLevel.MEMORY_ONLY_SER)
    val posTest = sc.parallelize(testHam).toDF("text", "label").persist(StorageLevel.MEMORY_ONLY_SER)
    val train = sc.parallelize(trainHam.union(trainSpan)).toDF("text", "label").persist(StorageLevel.MEMORY_ONLY_SER)
    val test = negTest.unionAll(posTest)

    //Serialize as Kyro object
    train.persist(StorageLevel.MEMORY_ONLY_SER)

    /*
    * Estimate size of RDD
    *
    * 1. dfs.block.size - The default value in Hadoop 2.0 is 128MB. In the local mode the corresponding parameter is fs.local.block.size (Default value 32MB). It defines the default partition size.
    * 2. numPartitions - The default value is 0 which effectively defaults to 1. This is a parameter you pass to the sc.textFile(inputPath,numPartitions) method. It is used to decrease the partition size (increase the number of partitions) from the default value defined by dfs.block.size
    * 3. mapreduce.input.fileinputformat.split.minsize - The default value is 1 byte. It is used to increase the partition size (decrease the number of partitions) from the default value defined by dfs.block.size
     */

//    println("fs.local.block.size: " + sc.getConf.get("fs.local.block.size"))
//    println("mapreduce.input.fileinputformat.split.minsize: " + sc.getConf.get("mapreduce.input.fileinputformat.split.minsize"))

    println("size of trainRDD: " + SizeEstimator.estimate(train) + " Byte")
    println("size of testRDD: " + SizeEstimator.estimate(test) + " Byte")

    train.repartition(1).show(3)

    train.repartition(1)

    println("构造流水线")

    //Tokenize text
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    //Transforms vectors of words into vectors of numeric codes for the purpose of further processing by NLP or machine learning algorithms.
    val word2vec = new Word2Vec()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")

    //Creates a logistic regression model.
    val lr = new LogisticRegression()

    //Create pipeline for tokenizer, word2vec, then lr
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, word2vec, lr))

    println("构造参数网络")
    /*    CrossValidator

        We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
        This will allow us to jointly choose parameters for all Pipeline stages.
        A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        We use a ParamGridBuilder to construct a grid of parameters to search over.
        With 3 values for word2vec.vectorSize (Dimension) and 3 values for lr.regParam (Regulation Parameter),
        this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
    */
    val paramGrid = new ParamGridBuilder()
      .addGrid(word2vec.vectorSize, Array(50, 100, 200))
      .addGrid(lr.regParam, Array(0.00001, 0.001, 0.1))
      .build()

    println("训练模型(包括Word2Vec特征抽取和LR分类模型)\n")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    println(cv.explainParams())

    benchmark("cross validation") {

      sqlContext.setConf("spark.sql.codegen.wholeStage", "true")

      model = cv.fit(train)

    }

    //val  model = cv.fit(train)

//    println(model.bestModel)
//
//    print(model.extractParamMap())

    val params = model.getEstimatorParamMaps
      .zip(model.avgMetrics)
      .maxBy(_._2)
      ._1

    println(params)

    println("评估模型结果:使用AUC指标")
    /*
    * Area under the curves (AUC) is the area below these ROC curves. Therefore, in other words, AUC is a great indicator of how well a classifier functions.
    * AUC is commonly used to compare the performance of various models while precision/recall/F-measure can help determine the appropriate threshold to use for prediction purposes.
    * */

    val testResult = model.transform(test)
    val evaluator = new BinaryClassificationEvaluator()
    println(s"test metrics: ${evaluator.evaluate(testResult)}")

    println("\n评估模型结果:负样本与正样本准确率")

    val negPrecision = precision(model, negTest)
    println(s"negative-precision = $negPrecision")

    val posPrecision = precision(model, posTest)
    println(s"positive-precision = $posPrecision")

  }

  def loadData(filePath: String): (Vector[(String, Double)], Vector[(String, Double)]) = {
    def reformat(sms: String, label: Double): (String, Double) = (sms.split("\t").last, label)

    //val file = Source.fromURL(filePath, "UTF-8").getLines().toVector.tail
    val file = Source.fromFile(filePath, "UTF-8").getLines().toVector.tail

    val (spam, ham) = file.partition(_.contains(CLASS_SPAM))
    val spamData = spam.map(x => reformat(x, LABEL_POSITIVE))
    val hamData = ham.map(x => reformat(x, LABEL_NEGATIVE))
    (spamData, hamData)
  }

  def precision(model: Model[_], test: DataFrame): Double = {

    val testResult = model.transform(test)

    testResult.head()

    val total = testResult.count()
    val corrects = testResult.filter("prediction = label").count()

    corrects.asInstanceOf[Double] / total
  }

  def benchmark(name: String)(f: => Unit )  = {
    val startTime = System.nanoTime
    f
    val endTime = System.nanoTime
    println(s"Time taken in $name: " + (endTime - startTime).toDouble / 1000000000 + " seconds")


  }

}
