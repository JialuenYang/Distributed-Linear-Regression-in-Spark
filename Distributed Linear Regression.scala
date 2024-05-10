// Databricks notebook source
// MAGIC %md
// MAGIC Implement closed-form solution when m(number of examples is large) and n(number of features) is small:
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]
// MAGIC Here, X is a distributed matrix.

// COMMAND ----------

// MAGIC %md
// MAGIC Steps:
// MAGIC 1. Create an example RDD for matrix X and vector y
// MAGIC 2. Compute \\[ \scriptsize \mathbf{(X^TX)}\\]
// MAGIC 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse
// MAGIC 4. Compute \\[ \scriptsize \mathbf{X^Ty}\\] and convert it to Breeze Vector
// MAGIC 5. Multiply \\[ \scriptsize \mathbf{(X^TX)}^{-1}\\] with \\[ \scriptsize \mathbf{X^Ty}\\]

// COMMAND ----------

// Jia-luen Yang

import breeze.linalg.{DenseMatrix, DenseVector, pinv, inv}

// COMMAND ----------

// Create a 4 x 3 matrix X and 4 vector y. Matrix X has a leftmost column of 1's.
val X = Array(Array(1.0, 1.0, -10.0), Array(1.0, 2.0, 10.0),  Array(1.0, 3.0, -10.0), Array(1.0, 4.0, 10.0))
val y = Array(3.0, 13.0, 23.0, 33.0)

val X_RDD = sc.parallelize(X.zipWithIndex.flatMap { case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value) } })

val y_RDD = sc.parallelize(y).zipWithIndex().map{ case (value, i) => (i.toInt, value) }

// COMMAND ----------

// X transpose
val X_T_RDD = X_RDD.map{ case ((i, j), value) => (i, (j, value)) }.cache()

// X.T matmult X
val XT_X = X_T_RDD.join(X_RDD.map{ case ((i, j), v) => (i, (j, v)) })
                  .map{ case (_, ((i, x), (j, y))) => ((i, j), x * y) }
                  .reduceByKey(_ + _)

// COMMAND ----------

// 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse

val n = XT_X.map(_._1._1).max + 1

val breezeDenseMatrix = DenseMatrix.zeros[Double](n, n)
XT_X.collect.foreach { case ((i, j), value) =>
      breezeDenseMatrix(i, j) = value
    }

val pseudoInverse = pinv(breezeDenseMatrix)

// COMMAND ----------

// XT multiply by y
val XT_y = X_T_RDD.join(y_RDD)
                  .map{ case (_, ((i, x), y)) => (i, x * y) }
                  .reduceByKey(_ + _)

// COMMAND ----------

val breezeDenseVector = DenseVector.zeros[Double](n)
XT_y.collect.foreach { case (i, value) =>
  breezeDenseVector(i) = value
}


// COMMAND ----------

val result: DenseVector[Double] = pseudoInverse * breezeDenseVector

// COMMAND ----------

// MAGIC %md
// MAGIC Implement \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\] where you compute \\[ \scriptsize \mathbf{\theta=(X^TX)}\\] using outer-product technique. 

// COMMAND ----------

// Create a 4 x 3 matrix X and 4 vector y. Matrix X has a leftmost column of 1's.
val X = Array(Array(1.0, 1.0, -10.0), Array(1.0, 2.0, 10.0),  Array(1.0, 3.0, -10.0), Array(1.0, 4.0, 10.0))
val y = Array(3.0, 13.0, 23.0, 33.0)

val X_RDD = sc.parallelize(X.zipWithIndex.flatMap { case (row, i) => row.zipWithIndex.map { case (value, j) => ((i.toInt, j.toInt), value) } })

val y_RDD = sc.parallelize(y).zipWithIndex().map{ case (value, i) => (i.toInt, value) }

val X_T_RDD = X_RDD.map{ case ((i, j), value) => (i, (j, value)) }.cache() // Cached for later use: X.T * y

val n = X_T_RDD.map(_._2._1).max + 1

// Uses a single reduceByKey
val XT_X = X_T_RDD.flatMap { case (i, (j, v)) =>
                    // Emit the value to all Result(i, j) that uses it to multiply
                    (for {
                      new_i <- 0 to j // Optimization to keep only above diagonal
                    } yield ((new_i, j), Array((i, (false, v))))) ++
                    (for {
                      new_j <- j to n-1 // Optimization to keep only above diagonal
                    } yield ((j, new_j), Array((i, (false, v)))))
                  }
                  .reduceByKey{(arr1, arr2) => 
                    val combinedArray = arr1 ++ arr2
                    val groupedMap = combinedArray.groupBy(_._1)
                    val multipliedArray = groupedMap.mapValues { pairs =>
                      // If there is only one element with the same key, keep the false and value.
                      if (pairs.length == 1) {
                        pairs.head._2
                      } 
                      else {
                        // Otherwise, multiply the 2 values and change flag to true
                        (true, pairs.map(_._2._2).product)
                      }
                    }
                    multipliedArray.toArray
                  }
                  .mapValues(_.map { case (_, (flag, value)) => if (flag) value else 0 }.sum)


// COMMAND ----------

val n = XT_X.map(_._1._1).max + 1

val breezeDenseMatrix = DenseMatrix.zeros[Double](n, n)
XT_X.collect.foreach { case ((i, j), value) =>
      breezeDenseMatrix(i, j) = value
      breezeDenseMatrix(j, i) = value // Optimization to keep only above diagonal
    }

val pseudoInverse = pinv(breezeDenseMatrix)

// COMMAND ----------

// X.T multiply by y
val XT_y = X_T_RDD.join(y_RDD)
                  .map{ case (_, ((i, x), y)) => (i, x * y) }
                  .reduceByKey(_ + _)

// COMMAND ----------

val breezeDenseVector = DenseVector.zeros[Double](n)
XT_y.collect.foreach { case (i, value) =>
  breezeDenseVector(i) = value
}

val result: DenseVector[Double] = pseudoInverse * breezeDenseVector

// COMMAND ----------

// MAGIC %md
// MAGIC Run algorithm on Boston Housing Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices?resource=download

// COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/tables/"))

// COMMAND ----------

// Parse CSV file
val housing_rdd = spark.sparkContext.textFile("dbfs:/FileStore/tables/housing.csv")
  .map(_.split("\\s+").filterNot(_.isEmpty))
  .map(_.map(_.toDouble))
  .zipWithIndex()
  .map{ case (row, index) => (index.toInt, Row.fromSeq(1.0 +: row)) } // Add leftmost column of 1's for intercept

// Load data into X_RDD and y_RDD
val X_RDD = housing_rdd.flatMap { case (index, row) =>
  row.toSeq.dropRight(1).zipWithIndex.map { case (value, columnIndex) =>
    ((index, columnIndex.toInt), value.asInstanceOf[Double]) 
  }
}

val y_RDD = housing_rdd.map { case (index, row) =>
  (index, row.getDouble(row.length-1)) 
}

// COMMAND ----------

// Base Normal Method

// X transpose
val X_T_RDD = X_RDD.map{ case ((i, j), value) => (i, (j, value)) }.cache()

// X.T matmult X
val XT_X = X_T_RDD.join(X_RDD.map{ case ((i, j), v) => (i, (j, v)) })
                  .filter { case (k, ((i, x), (j, y))) => i <= j }  // Optimization to keep only above diagonal
                  .map{ case (_, ((i, x), (j, y))) => ((i, j), x * y) }
                  .reduceByKey(_ + _)

val n = XT_X.map(_._1._1).max + 1

val breezeDenseMatrix = DenseMatrix.zeros[Double](n, n)
XT_X.collect.foreach { case ((i, j), value) =>
      breezeDenseMatrix(i, j) = value
      breezeDenseMatrix(j, i) = value // Optimization to keep only above diagonal
    }

val pseudoInverse = pinv(breezeDenseMatrix)

// XT multiply by y
val XT_y = X_T_RDD.join(y_RDD)
                  .map{ case (_, ((i, x), y)) => (i, x * y) }
                  .reduceByKey(_ + _)

val breezeDenseVector = DenseVector.zeros[Double](n)
XT_y.collect.foreach { case (i, value) =>
  breezeDenseVector(i) = value
}

val result_normal: DenseVector[Double] = pseudoInverse * breezeDenseVector


// COMMAND ----------

// Bonus 1 Method

val X_T_RDD = X_RDD.map{ case ((i, j), value) => (i, (j, value)) }.cache() // Cached for later use: X.T * y

val n = X_T_RDD.map(_._2._1).max + 1

val XT_X = X_T_RDD.flatMap { case (i, (j, v)) =>
                    // Emit the value to all Result(i, j) that uses it to multiply
                    (for {
                      new_i <- 0 to j // Optimization to keep only above diagonal
                    } yield ((new_i, j), Array((i, (false, v))))) ++
                    (for {
                      new_j <- j to n-1 // Optimization to keep only above diagonal
                    } yield ((j, new_j), Array((i, (false, v)))))
                  }
                  .reduceByKey{(arr1, arr2) => 
                    val combinedArray = arr1 ++ arr2
                    val groupedMap = combinedArray.groupBy(_._1)
                    val multipliedArray = groupedMap.mapValues { pairs =>
                      // If there is only one element with the same key, keep the false and value.
                      if (pairs.length == 1) {
                        pairs.head._2
                      } 
                      else {
                        // Otherwise, multiply the 2 values and change flag to true
                        (true, pairs.map(_._2._2).product)
                      }
                    }
                    multipliedArray.toArray
                  }
                  .mapValues(_.map { case (_, (flag, value)) => if (flag) value else 0 }.sum)

val breezeDenseMatrix = DenseMatrix.zeros[Double](n, n)
XT_X.collect.foreach { case ((i, j), value) =>
      breezeDenseMatrix(i, j) = value
      breezeDenseMatrix(j, i) = value // Optimization to keep only above diagonal
    }

val pseudoInverse = pinv(breezeDenseMatrix)

// X.T multiply by y
val XT_y = X_T_RDD.join(y_RDD)
                  .map{ case (_, ((i, x), y)) => (i, x * y) }
                  .reduceByKey(_ + _)

val breezeDenseVector = DenseVector.zeros[Double](n)
XT_y.collect.foreach { case (i, value) =>
  breezeDenseVector(i) = value
}

val result_bonus1: DenseVector[Double] = pseudoInverse * breezeDenseVector

// COMMAND ----------

// Proving both methods work by predicting the y value of first row of X

val firstLineArray = Array(1.0) ++ spark.sparkContext.textFile("dbfs:/FileStore/tables/housing.csv")
                                                   .first()
                                                   .split("\\s+")
                                                   .filterNot(_.isEmpty)
                                                   .map(_.toDouble)

val x_array = firstLineArray.dropRight(1)
val true_y = firstLineArray(firstLineArray.length - 1)

// Test the first row of housing data
val normal_predict = result_normal.toArray.zip(x_array).map { case (x, y) => x * y }.sum
val bonus1_predict = result_bonus1.toArray.zip(x_array).map { case (x, y) => x * y }.sum

// COMMAND ----------

// MAGIC %md
// MAGIC Both predictions are around 30. True y value is 24. That's close enough.
