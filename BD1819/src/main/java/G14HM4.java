import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.BLAS;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class G14HM4
{
    public static void main(String[] args) throws Exception
    {

        //------- PARSING CMD LINE ------------
        // Parameters are:
        // <path to file>, k, L and iter

        if (args.length != 4) {
            System.err.println("USAGE: <filepath> k L iter");
            System.exit(1);
        }
        String inputPath = args[0];
        int k=0, L=0, iter=0;
        try
        {
            k = Integer.parseInt(args[1]);
            L = Integer.parseInt(args[2]);
            iter = Integer.parseInt(args[3]);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
        if(k<=2 && L<=1 && iter <= 0)
        {
            System.err.println("Something wrong here...!");
            System.exit(1);
        }
        //------------------------------------
        final int k_fin = k;

        //------- DISABLE LOG MESSAGES
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //------- SETTING THE SPARK CONTEXT
        SparkConf conf = new SparkConf(true).setAppName("kmedian new approach");
        JavaSparkContext sc = new JavaSparkContext(conf);

        //------- PARSING INPUT FILE ------------
        JavaRDD<Vector> pointset = sc.textFile(args[0], L)
                .map(x-> strToVector(x))
                .repartition(L)
                .cache();
        long N = pointset.count();
        System.out.println("Number of points is : " + N);
        System.out.println("Number of clusters is : " + k);
        System.out.println("Number of parts is : " + L);
        System.out.println("Number of iterations is : " + iter);

        //------- SOLVING THE PROBLEM ------------
        double obj = MR_kmedian(pointset, k, L, iter);
        System.out.println("Objective function is : <" + obj + ">");
    }

    public static Double MR_kmedian(JavaRDD<Vector> pointset, int k, int L, int iter)
    {
        //
        // --- ADD INSTRUCTIONS TO TAKE AND PRINT TIMES OF ROUNDS 1, 2 and 3
        //
        long start, end;

        //------------- ROUND 1 ---------------------------
        start = System.currentTimeMillis();
        JavaRDD<Tuple2<Vector,Long>> coreset = pointset.mapPartitions(x ->
        {
            ArrayList<Vector> points = new ArrayList<>();
            ArrayList<Long> weights = new ArrayList<>();
            while (x.hasNext())
            {
                points.add(x.next());
                weights.add(1L);
            }
            ArrayList<Vector> centers = kmeansPP(points, weights, k, iter);
            ArrayList<Long> weight_centers = compute_weights(points, centers);
            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weight_centers.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        });

        //------------- ROUND 2 ---------------------------
        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>(k*L);
        elems.addAll(coreset.collect());
        end = System.currentTimeMillis();
        System.out.println("Execution time of Round 1:  "+(end-start)+"ms");
        start = System.currentTimeMillis();
        ArrayList<Vector> coresetPoints = new ArrayList<>();
        ArrayList<Long> weights = new ArrayList<>();
        for(int i =0; i< elems.size(); ++i)
        {
            coresetPoints.add(i, elems.get(i)._1);
            weights.add(i, elems.get(i)._2);
        }

        ArrayList<Vector> centers = kmeansPP(coresetPoints, weights, k, iter);
        end = System.currentTimeMillis();
        System.out.println("Execution time of Round 2:  "+(end-start)+"ms");

        //------------- ROUND 3: COMPUTE OBJ FUNCTION --------------------
        start = System.currentTimeMillis();
        double objective_funtion = pointset.map(x->{

            double best_dist = euclidean(centers.get(0), x);
            for (int j = 1; j < k; j++)
            {
                double dist = euclidean(centers.get(j), x);

                if (dist < best_dist)
                {
                    best_dist = dist;
                }
            }
            return best_dist;
        }).reduce((x,y)-> x+y);
        end = System.currentTimeMillis();

        System.out.println("Execution time of Round 3:  "+(end-start)+"ms");
        return objective_funtion/(pointset.count());
    }

    public static ArrayList<Long> compute_weights(ArrayList<Vector> points, ArrayList<Vector> centers)
    {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i =0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j)
            {
                if(euclidean(points.get(i),centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // Euclidean distance
    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }

    /**
     * Computation of clustering using
     * ----First a variant of kmeans++ (weighted)
     * ----Second a particular implementation of Lloyd's algorithm
     * @param P input dataset
     * @param WP weights of points in P
     * @param k number of clusters that we want to generate
     * @param iter number of iterations of Lloyd's algorithm
     * @return the set of k centroids of the clustering
     */
    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Long> WP, int k, int iter)
    {
        int size_P = P.size();

        boolean [] isCenter = new boolean[size_P];

        for(int i=0; i<size_P; isCenter[i++]=false);

        /**
         * KMEANS++  VARIATION
         */
        //generator of randomic numbers
        Random random = new Random();

        //index of first center, chosen randomly from the dataset P
        int index = random.nextInt(size_P);

        //Addition of first element, chosen randomly in the previous step, to centers set
        ArrayList<Vector> centers = new ArrayList<>();
        centers.add(P.get(index));

        //last center added to centers set
        Vector last_add = P.get(index);
        isCenter[index]=true;

        //distances of points in P, that weren't chosen as center, from the previous computed center
        ArrayList<Double> distances = new ArrayList<>();
        for(int i=0; i<size_P; i++)
        {
            //initialization of distance looking to the single center in centers set
            distances.add(Math.sqrt(Vectors.sqdist(last_add, P.get(i))));
        }

        for (int iteration = 1; iteration < k ; iteration++)
        {
            double sum = 0;

            //Generation of a random value between 0 and 1
            double prob = random.nextDouble();

            index=0;

            double min=0;
            boolean flag=false;

            for(int i=0; i< size_P; i++)
            {
                sum += (WP.get(i) * distances.get(i));
            }
            //Choice of next center looking to the weighted "probability"
            //of each point in the dataset and the random number previously generated

            for(int i=0; i < size_P && !flag; i++)
            {
                double max = min + ((WP.get(i) * distances.get(i)) / sum);

                if (prob != 0)
                {
                    if (prob <= max)
                    {
                        index = i;
                        flag = true;
                    }
                    else
                        min = max;
                }
            }

            //Update of centers set, P set and WP set (storing last added center)
            last_add = P.get(index);
            centers.add(last_add);
            isCenter[index]=true;
            distances.set(index,0.0);

            //Computation of distances for each point from the last added center
            //And update of it
            for (int i = 0; i < size_P; i++)
            {
                //The computation is done if and only if a point wasn't chosen as a center
                if(!isCenter[i])
                {
                    double dist = Math.sqrt(Vectors.sqdist(last_add, P.get(i)));

                    if (dist < distances.get(i))
                        distances.set(i, dist);
                }
            }


        }

        //Partition of initial dataset using previously computed centers
        ArrayList<ArrayList<Integer>> partition = partition(P, centers);

        /**
         * LLOYD'S  ALGORITHM
         */
        //Creation of set of k centroids (initially with all coordinates equal to 0)
        //only if the Lloyd's algorithm can be applied (iter!=0)
        if(iter==0)
            return centers;

        ArrayList<Vector> centroids = new ArrayList<>();

        for (int i=0; i<k; i++)
            centroids.add(Vectors.zeros(P.get(0).size()));

        //Computation of Lloyd's algorithm for iter iterations
        for (int i = 0; i < iter; i++)
        {
            //Define the centroid of each partition
            for (int j = 0; j < k; j++)
            {
                ArrayList<Integer> cluster = partition.get(j);
                int size = cluster.size();
                long sum_WP=0;

                //Computation of component-wise sum of points that belong to
                //the same cluster and sum of all their weights
                for (int y=0; y<size; y++)
                {
                    BLAS.axpy(WP.get(cluster.get(y)), P.get(cluster.get(y)), centroids.get(j));
                    sum_WP+=WP.get(cluster.get(y));
                }

                //Computation of a centroid dividing the component-wise sum
                //by the previously computed sum of weights
                BLAS.scal(1.0 / sum_WP, centroids.get(j));
            }
            //Partition of initial dataset using new centroids
            partition = partition(P, centroids);
        }

        return centroids;
    }

    /**
     * Computation of partition of a dataset P from a dataset of centers/centroids
     * @param P set of points
     * @param centers set of points chosen as centers
     * @return set of k clusters (each cluster is defined by the indexes of points in P that belongs to it)
     */
    private static ArrayList<ArrayList<Integer>> partition(ArrayList<Vector> P, ArrayList<Vector> centers) {
        int k = centers.size();

        //Creation of the set of k clusters
        ArrayList<ArrayList<Integer>> partition = new ArrayList<>();
        for (int j = 0; j < k; j++)
            partition.add(new ArrayList<>());

        //Define to which cluster each point of P belongs
        for (int i = 0; i < P.size(); i++) {
            double best_dist = Math.sqrt(Vectors.sqdist(centers.get(0), P.get(i)));
            int cluster_index = 0;
            for (int j = 1; j < k; j++) {
                double dist = Math.sqrt(Vectors.sqdist(centers.get(j), P.get(i)));

                if (dist < best_dist) {
                    best_dist = dist;
                    cluster_index = j;
                }
            }
            //addition of index i of i-th point of P to the cluster of index cluster_index with closest center to it
            partition.get(cluster_index).add(i);
        }

        return partition;
    }
}
