import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;

import scala.Tuple2;

import java.util.*;


public class G13HM2
{
    public static void main(String[] args)
    {
        int i_partitions = -1;
        /* Input check. */
        try
        {
            i_partitions = Integer.parseInt(args[0]);
        }
        catch(NumberFormatException e)
        {
            System.out.println("Insert an integer.");
            return;
        }

        /* Creation of Spark configuration and context. */
        SparkConf configuration = new SparkConf(true)
                        .setAppName("application name here");
        JavaSparkContext sc = new JavaSparkContext(configuration);

        /* Dataset input. */
        JavaRDD<String> documentsRDD = sc.textFile(args[1]);
        JavaRDD<String> partitionedDocsRDD = documentsRDD.repartition(i_partitions); /* Dataset partitioning. */
        documentsRDD.cache();
        documentsRDD.count();
        partitionedDocsRDD.cache();
        partitionedDocsRDD.count();

        /* Improved Word Count 1 */
        long startTime = System.currentTimeMillis();
        JavaPairRDD<String, Long> count1RDD = improvedWordCount1(documentsRDD);
        count1RDD.cache();
        long numberOfWords = count1RDD.count(); /* Number of individual words in the document. */
        long endTime = System.currentTimeMillis();
        System.out.println("The improved Word Count 1 takes " + (endTime - startTime) + "ms");

        /* First Improved Word Count 2 */
        startTime = System.currentTimeMillis();
        JavaPairRDD<String, Long> count2aRDD = improvedWordCount2a(documentsRDD, i_partitions);
        count2aRDD.cache();
        count2aRDD.count();
        endTime = System.currentTimeMillis();
        System.out.println("The first Improved Word Count 2 takes " + (endTime - startTime) + "ms");

        /*Second Improved Word Count 2 */
        startTime = System.currentTimeMillis();
        JavaPairRDD<String, Long> count2bRDD = improvedWordCount2b(partitionedDocsRDD, i_partitions);
        count2bRDD.cache();
        count2bRDD.count();
        endTime = System.currentTimeMillis();
        System.out.println("The second Improved Word Count 2 takes " + (endTime - startTime) + "ms");


        JavaPairRDD<String, Long> strLenRDD = count1RDD.mapToPair((x) ->
        {
            long l = x._1.length();
            return new Tuple2<>(x._1, l);
        });

        Tuple2<String, Long> strLenSum = strLenRDD.reduce((x, y) ->
        {
            long l = x._2 + y._2;
            return new Tuple2<>("a", l);
        });
        double averageLen = (double)strLenSum._2/numberOfWords;
        System.out.println("The average length of distinct words is: " + averageLen);

        /* Print all elements in an RDD. */
        /*
        for (Tuple2<String, Long> element : count2aRDD.collect())
        {
            System.out.println(element._1() + " " + element._2());
        }
        */

        try
        {
            System.in.read();
        }
        catch(java.io.IOException e)
        {

        }


    }

    /* Improved Word Count 1 */
    public static JavaPairRDD<String,Long> improvedWordCount1(JavaRDD<String> documentsRDD)
    {
        JavaPairRDD<String, Long> docRDD = documentsRDD
                // Map phase
                .flatMapToPair((document) ->
                {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (String token : tokens)
                    {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }
                    for (Map.Entry<String, Long> e : counts.entrySet())
                    {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce phase
                .reduceByKey((x,y) -> x + y);

        return docRDD;
    }


    /* First Improved Word Count 2 */
    public static JavaPairRDD<String,Long> improvedWordCount2a(JavaRDD<String> documentsRDD, int k)
    {
        Random randomGenerator = new Random(); /* Random number generator. */

        JavaPairRDD<String, Long> docRDD = documentsRDD
            /*Round 1*/
            /* Map Phase */
            .flatMapToPair((document) ->
            {
                String[] tokens = document.split(" ");
                HashMap<String, Long> counts = new HashMap<>();
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                for (String token : tokens)
                {
                    counts.put(token, 1L + counts.getOrDefault(token, 0L));
                }
                for (Map.Entry<String, Long> e : counts.entrySet())
                {
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                return pairs.iterator();
            })
            .groupBy((x) ->
            {
                long randomInt = randomGenerator.nextInt(k);
                return randomInt;
            }, k)
            /* Reduce Phase*/
            .flatMapToPair((x) ->
            {
                HashMap<String, Long> counts = new HashMap<>();
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                Iterator<Tuple2<String, Long>> list = x._2().iterator();
                while (list.hasNext())
                {
                    Tuple2<String, Long> element = list.next();
                    counts.put(element._1(), element._2() + counts.getOrDefault(element._1(), 0L));
                }

                for (Map.Entry<String, Long> e : counts.entrySet())
                {
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                return pairs.iterator();
            })
            /* Round 2 */
            /* Map Phase - Identity */
            /* Reduce Phase */
            .reduceByKey((x,y) -> x + y);
        return docRDD;
    }


    /* Second improved Word Count 2*/
    public static JavaPairRDD<String,Long> improvedWordCount2b(JavaRDD<String> partitionedDocsRDD, int i_partitions)
    {
        /*Round 1*/
        /* Map Phase */
        JavaPairRDD<String, Long> docRDD = partitionedDocsRDD.flatMapToPair((document) ->
        {
            String[] tokens = document.split(" ");
            HashMap<String, Long> counts = new HashMap<>();
            ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
            for (String token : tokens)
            {
                counts.put(token, 1L + counts.getOrDefault(token, 0L));
            }
            for (Map.Entry<String, Long> e : counts.entrySet())
            {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
            return pairs.iterator();
        });

        /* Reduce Phase */
        JavaPairRDD<String, Long> partialCountRDD = docRDD.mapPartitionsToPair((x) ->
        {
            HashMap<String, Long> counts = new HashMap<>();
            ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
            while (x.hasNext())
            {
                Tuple2<String, Long> element = x.next();
                counts.put(element._1(), element._2() + counts.getOrDefault(element._1(), 0L));
            }

            for (Map.Entry<String, Long> e : counts.entrySet())
            {
                pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
            }
            return pairs.iterator();
        });

        /* Round 2 */
        /* Map Phase - Identity */
        /* Reduce Phase */
        JavaPairRDD<String, Long> finalCountRDD = partialCountRDD.reduceByKey((x,y) -> x + y);
        return finalCountRDD;
    }
}


