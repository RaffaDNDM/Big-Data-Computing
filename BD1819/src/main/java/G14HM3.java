import java.util.*;
import java.lang.String;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.BLAS;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class G14HM3
{

    public static void main(String[] args)
    {
        /*
            in input, we want to have 3 parameters on command line at execution time
            args[0]=name of the file
            args[1]=number of clusters that we want to produce
            args[2]=number of iterations for Lloyd's algorithm
         */
        if (args.length != 3)
        {
            System.out.println("Error specifying parameters in execution time");
        }

        //input = vector of input points
        ArrayList<Vector> input=null;
        try
        {
            input = readVectorsSeq(args[0]);
        }
        catch (IOException e) {
            e.printStackTrace();
        }


        int size=input.size();
        //Initialization of vector of weights for all the points in input
        ArrayList<Integer> WP= new ArrayList<>();

        for (int i=0; i<size; i++)
        {   //we assume that all the weights are equal to 1
            WP.add(1);
        }

        //Computation of clustering
        ArrayList<Vector> cluster_k=kmeansPP(input,WP, Integer.parseInt(args[1]),Integer.parseInt(args[2]));
        //Average distance of a point from its center
        System.out.println(kmeansObj(input,cluster_k));
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
    public static ArrayList<Vector> kmeansPP(ArrayList<Vector> P, ArrayList<Integer> WP, int k, int iter)
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
                int sum_WP=0;

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
    private static ArrayList<ArrayList<Integer>> partition(ArrayList<Vector> P, ArrayList<Vector> centers)
    {
        int k = centers.size();

        //Creation of the set of k clusters
        ArrayList<ArrayList<Integer>> partition = new ArrayList<>();
        for (int j = 0; j < k; j++)
            partition.add(new ArrayList<>());

        //Define to which cluster each point of P belongs
        for (int i = 0; i < P.size(); i++)
        {
            double best_dist = Math.sqrt(Vectors.sqdist(centers.get(0), P.get(i)));
            int cluster_index = 0;
            for (int j = 1; j < k; j++)
            {
                double dist = Math.sqrt(Vectors.sqdist(centers.get(j), P.get(i)));

                if (dist < best_dist)
                {
                    best_dist = dist;
                    cluster_index = j;
                }
            }
            //addition of index i of i-th point of P to the cluster of index cluster_index with closest center to it
            partition.get(cluster_index).add(i);
        }

        return partition;
    }

    /**
     * Compute the average of the distances of the points from their center
     * @param P dataset of points
     * @param C set of centers
     * @return average of the distances of the points from their centers
     */
    public static double kmeansObj(ArrayList<Vector> P, ArrayList<Vector> C)
    {
        int size= P.size();
        int k = C.size();
        double average=0;

	//compute the distances of points from the center of their cluster
	//and sum all of them
        for (int i = 0; i < P.size(); i++)
        {
            double best_dist = Math.sqrt(Vectors.sqdist(C.get(0), P.get(i)));

            for (int j = 1; j < k; j++)
            {
                double dist =Math.sqrt(Vectors.sqdist(C.get(j), P.get(i)));

                if (dist < best_dist)
                {
                    best_dist = dist;
                }
            }
            average+=best_dist;
        }

	    //return the average of the distances (dividing the sum by the size of the dataset)
        return average/size;
    }

    private static Vector strToVector(String str) {
        String[] tokens = str.split(" ");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    private static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }
}
