import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Dataset, SparkSession, Row}

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

object Recommender {

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]) {

    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    import spark.implicits._

    val ratings = spark.read.textFile("/Users/soft/repos/mingle/mingle_recommender/data/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()

    val split: Array[Dataset[Row]] = ratings.randomSplit(Array(0.8, 0.2))

    val training = split(0)
    val test = split(1)
    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(training)
  }
}