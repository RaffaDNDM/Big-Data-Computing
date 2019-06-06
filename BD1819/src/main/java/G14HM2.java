import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import java.util.*;
import java.lang.String;

public class G14HM2 {

    public static void main (String [] args)
    {
        if (args.length != 2)
        {
            throw new IllegalArgumentException("Expecting file name and number of partition on command line");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework");
        JavaSparkContext sc = new JavaSparkContext(conf);

        /**
         * args[0]= k
         * args[1]= filename
         */
        // RDD composed of strings (one string = one document)
        JavaRDD<String> docs = sc.textFile(args[1]);
        // RDD with the same strings as docs but partitioned with Spark method
        JavaRDD<String> partioned_docs = docs.repartition(Integer.parseInt(args[0]));
        docs.cache();
        docs.count();

        //ImprovedWordCount1
        long start = System.currentTimeMillis();
        JavaPairRDD<String, Long> words_Iwc1 = improvedWordCount1(docs);
        long end = System.currentTimeMillis();

        System.out.println("Execution time of Improved Word Count 1:  "+(end-start)+"ms");

        //ImprovedWordCount2 with random keys
        start = System.currentTimeMillis();
        JavaPairRDD<String, Long> words_Iwc2_random = improvedWordCount2_random(docs,Integer.parseInt(args[0]));
        end = System.currentTimeMillis();

        System.out.println("Execution time of Improved Word Count 2 with random keys" + ":  "+(end-start)+"ms");

        //ImprovedWordCount2 with Spark partition
        start = System.currentTimeMillis();
        JavaPairRDD<String, Long> words_Iwc2_Spark = improvedWordCount2_Spark(partioned_docs);
        end = System.currentTimeMillis();

        System.out.println("Execution time of Improved Word Count 2 with Spark partition " + ":  "+(end-start)+"ms");

        System.out.println("The average length of the distinct words is:  "+ average(words_Iwc1));
    }


    public static JavaPairRDD<String, Long> improvedWordCount1 (JavaRDD<String> docs)
    {
        JavaPairRDD<String, Long> wordcountpairs = docs
                // Map phase
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    //Occurrences count in a document
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    //Creation of HashMap containing all the different words of the document (with their occurrences)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce phase
                // Count of total number of occurrences for each different word in all the documents
                .reduceByKey((a, b) -> a + b);

        wordcountpairs.cache();
        wordcountpairs.count();
        return wordcountpairs;
    }


    public static JavaPairRDD<String, Long> improvedWordCount2_random (JavaRDD<String> docs, int k)
    {
        JavaPairRDD<String,Long> partition = docs
                // **************     Round 1      ***********************
                // Map phase
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    //Occurrences count in a document
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    //Creation of HashMap containing all the different words of the document (with their occurrences)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Associates to each pair a random key in the range [0,k-1]
                .groupBy(x -> {
                    Random num = new Random();
                    long value = num.nextLong() % ( (long) k);

                    return value;
                })
                // Reduce phase
                // Count of total number of occurrences for each different word
                // between words with the same random key
                .flatMapToPair((couples) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    //Return an iterator of pairs (string, #occurences) with the same random key
                    Iterator<Tuple2<String, Long>> set = couples._2().iterator();

                    //Occurrences count in the set of pairs previously returned (set)
                    while(set.hasNext())
                    {
                        Tuple2<String,Long> x = set.next();
                        counts.put(x._1(), x._2() + counts.getOrDefault(x._1(), 0L));
                    }

                    //Creation of HashMap containing all the different words of the set (with their occurrences)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })

                // **************     Round 2      ***********************
                // Map phase: identity
                // Reduce phase
                // For each different word, we compute the total number of occurrences
                .reduceByKey((x,y)->x+y);

        partition.cache();
        partition.count();

        return partition;
    }

    public static JavaPairRDD<String, Long> improvedWordCount2_Spark (JavaRDD<String> docs)
    {
        JavaPairRDD<String,Long> partition = docs
                // **************     Round 1      ***********************
                // Map phase
                .flatMapToPair((document) -> {
                    String[] tokens = document.split(" ");
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    //Occurrences count in a document
                    for (String token : tokens) {
                        counts.put(token, 1L + counts.getOrDefault(token, 0L));
                    }

                    //Creation of HashMap containing all the different words of the document (with their occurrences)
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // Reduce phase
                // Computes number of occurrences for each different word in the same partition
                .mapPartitionsToPair((couples) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();

                    // Check if there is already the pair (word,#occurences) in pairs
                    // and update the number of occurences
                    while(couples.hasNext())
                    {
                        Tuple2<String,Long> x = couples.next();
                        counts.put(x._1(), x._2() + counts.getOrDefault(x._1(), 0L));
                    }

                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                // **************     Round 2      ***********************
                // Map phase: identity
                // Reduce phase
                // For each different word, we compute the total number of occurrences
                .reduceByKey((x,y) -> x + y );

        partition.cache();
        partition.count();
        return partition;
    }

    public static double average (JavaPairRDD<String,Long> words)
    {
        //We keep only different words without the number of occurrences
        JavaRDD<String> occurrences = words.keys();

        // Map phase: x --> length of x
        // Reduce phase: we sum all the lengths of the words
        int sum = occurrences.map((x)-> x.length()).reduce((x,y)->x+y);

        // We divide sum by the number of different words
        return ((double) sum)/((double) occurrences.count());
    }

}
