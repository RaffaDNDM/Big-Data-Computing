import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Comparator;
import java.io.Serializable;

public class G14HM1
{
    public static void main(String[] args) throws FileNotFoundException
    {
        if (args.length == 0)
        {
            throw new IllegalArgumentException("Expecting the file name on the command line");
        }

        ArrayList<Double> lNumbers = new ArrayList<>();
        Scanner s =  new Scanner(new File(args[0]));
        while (s.hasNext())
        {
            lNumbers.add(Double.parseDouble(s.next()));
        }
        s.close();

        // Setup Spark
        SparkConf conf = new SparkConf(true).setAppName("Preliminaries");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a parallel collection
        JavaRDD<Double> dNumbers = sc.parallelize(lNumbers);

        /**
         * Computation of max through the reduce function
         */
         double max1 = dNumbers.reduce((x, y) -> {
            return (y>x)?y:x;
        });

        System.out.println("The max number with reduce is " + max1);

        /**
         * Computation of max through max function of Spark
         */
         double max2 = dNumbers.max(new Compare());

        System.out.println("The max number with max is " + max2);

        /**
         * Normalization of input data creating new RDD
         */
        JavaRDD<Double> dNormalized = dNumbers.map((x) -> x/max1);


        /**
         * Computation of probability based on number of occurrences
         * Using an approach similar to WordCount
         */
        long n= dNormalized.count();
        System.out.println("The probabilities of normalized values are");
        JavaPairRDD<Double, Integer> pairs = dNormalized.mapToPair(x -> new Tuple2<>(x, 1));
        JavaPairRDD<Double, Integer> counts = pairs.reduceByKey((a, b) -> a + b);
        counts.foreach(x->{
            double num=(double) x._2();

            System.out.println("Value = "+x._1()+"    Probability = "+ num/n);
        });
    }


    private static class Compare implements Serializable, Comparator<Double>
    {
        public int compare(Double a, Double b) {
            if (a > b)
                return 1;
            if (a < b)
                return -1;

            return 0;
        }
    }
}
