package com.reviewanalysis

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

class Classifier(filepath: String, sqlContext: SQLContext) {
  def splitPosNeg(review: Double) = if (review>3) 1 else 0
  val df = sqlContext.read.json(filepath)
  val tokenizer = new Tokenizer()
    .setInputCol("reviewText")
    .setOutputCol("tokens")
  val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol("tokens")
    .setOutputCol("features")
  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(.001)
  val pipeline = new Pipeline()
    .setStages(Array(hashingTF, lr))
  val model = pipeline.fit(df)
  
  def transform(dataset: Dataset[_]) = model.transform(dataset)
  
  
  
  
  
  
}