package com.bdu

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec}
import org.apache.spark.ml.tuning.{CrossValidatorModel, CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source


/**
 * Created by Eric on 8/11/16.
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

  def main(args: Array[String]): Unit = {

    println("初始化SQLContext")
    val conf = new SparkConf().setAppName("CrossValidation Pipeline").setMaster("local[*]") //Using 2 core running on local

    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = SQLContext.getOrCreate(sc)
    import sqlContext.implicits._
    sqlContext.setConf("spark.sql.codegen.wholeStage", "true")
    sqlContext.setConf("spark.executor.instances", "4")

    /*
    * Split training set, 80% for training, 20% for test.
    * */

    val (spam, ham) = loadData(TRAIN_DATA)
    val (trainSpan, testSpan) = spam.partition(_._1.length % 10 < 8)
    val (trainHam, testHam) = ham.partition(_._1.length % 10 < 8)


    val negTest = sc.parallelize(testSpan).toDF("text", "label").cache()
    val posTest = sc.parallelize(testHam).toDF("text", "label").cache()
    val train = sc.parallelize(trainHam.union(trainSpan)).toDF("text", "label").cache()
    val test = negTest.unionAll(posTest)

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
        With 3 values for word2vec.vectorSize and 3 values for lr.regParam,
        this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
    */
    val paramGrid = new ParamGridBuilder()
      .addGrid(word2vec.vectorSize, Array(50, 200))
      .addGrid(lr.regParam, Array(0.00001, 0.001, 0.1))
      .build()

    println("训练模型(包括Word2Vec特征抽取和LR分类模型)")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)



//    benchmark("cross validation") {
//
//      sqlContext.setConf("spark.sql.codegen.wholeStage", "true")
//
//      val m = cv.fit(train)
//      println(m.bestModel.params)
//
//    }

    val  model = cv.fit(train)

    println(model.bestModel)

    println("评估模型结果:使用AUC指标")
    /*
    * Area under the curves (AUC) is the area below these ROC curves. Therefore, in other words, AUC is a great indicator of how well a classifier functions.
    * AUC is commonly used to compare the performance of various models while precision/recall/F-measure can help determine the appropriate threshold to use for prediction purposes.
    * */

    val testResult = model.transform(test)
    val evaluator = new BinaryClassificationEvaluator()
    println(s"test metrics: ${evaluator.evaluate(testResult)}")

    println("评估模型结果:负样本与正样本准确率")

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

  def benchmark(name: String)(f: => Unit )  {
    val startTime = System.nanoTime
    f
    val endTime = System.nanoTime
    println(s"Time taken in $name: " + (endTime - startTime).toDouble / 1000000000 + " seconds")
  }

}
