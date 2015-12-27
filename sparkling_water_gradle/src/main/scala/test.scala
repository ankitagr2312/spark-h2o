import java.io.File

import hex.tree.gbm.GBM
import hex.tree.gbm.GBMModel.GBMParameters
import org.apache.spark.h2o.{StringHolder, H2OContext, H2OFrame}
import org.apache.spark.{SparkFiles, SparkConf, SparkContext}

object test
{
  def main(args: Array[String]) {
    val conf=new SparkConf().setMaster("local[*]").setAppName("Sparkling water gradle")
    val sc = new SparkContext(conf)

    // Create H2O Context
    val h2oContext = new H2OContext(sc).start()
    import h2oContext._

    // Register file to be available on all nodes
    sc.addFile(new File("data/prostate.csv").getAbsolutePath)

    // Load data and parse it via h2o parser
    val irisTable = new H2OFrame(new File(SparkFiles.get("prostate.csv")))

    val gbmParams = new GBMParameters()
    gbmParams._train = irisTable
    gbmParams._response_column = 'GLEASON
    gbmParams._ntrees = 5

    val gbm = new GBM(gbmParams)
    val gbmModel = gbm.trainModel.get


    val predict = gbmModel.score(irisTable)('predict)

    // Compute number of mispredictions with help of Spark API
    val trainRDD = asRDD[StringHolder](irisTable('GLEASON))
    val predictRDD = asRDD[StringHolder](predict)
  }
}