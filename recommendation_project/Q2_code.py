import pyspark

from pyspark.sql import SparkSession


spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()

# spark = SparkSession.builder \
#     .master("local[2]") \
#     .config("spark.local.dir","/fastdata/act18mc") \ #replace username with your own username
#     .appName("COM6012 Collaborative Filtering RecSys") \
#     .getOrCreate()

sc = spark.sparkContext

# Question2
# 2.A
# read ratings.csv
ratings = spark.read.load("../Data/ratings.csv", format="csv", header=True, inferSchema=True)

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

# choose 3 kinds of als with different parameters
als1 = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
als2 = ALS(maxIter=10, regParam=0.2, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
als3 = ALS(maxIter=20, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

# split into 3 ratings which means 3 fold
(rating1, rating2, rating3) = ratings.randomSplit([1/3, 1/3, 1/3], 190)

ratings_splits = [rating1, rating2, rating3]
# # generate 3 train data sets
train1 = rating1.union(rating2)
train2 = rating2.union(rating3)
train3 = rating1.union(rating3)

train_list = [train1, train2, train3]
test_list = [rating3, rating2, rating1]
als_list = [als1, als2, als3]

rmse_outter = [] # put all the rmse in 'rmse_outter'
mae_outter = [] # put all the mae in 'mae_outter'
model_list = [] # put all the train model in 'model_list' which will be used in the second child question
for als in als_list:
    rmse_inner = []
    mae_inner = []
    for i, train in enumerate(train_list):
        # train the model
        model = als.fit(train)
        model_list.append(model)
        predictions = model.transform(test_list[i])
        evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
        evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
        # get rmse and mae
        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        # save rmse and mae in one fold
        rmse_inner.append(rmse)
        mae_inner.append(mae)
        print(rmse)
        print(mae)
    # save all the rmse and mae
    rmse_outter.append(rmse_inner)
    mae_outter.append(mae_inner)

error_list = [] # error_list is used to save data part of a dataframe
# put each fold's corresponding rmse and mae in the same row
for i in range(3):
    error_list.append((rmse_outter[0][i],rmse_outter[1][i],rmse_outter[2][i],mae_outter[0][i],mae_outter[1][i],mae_outter[2][i]))

# generate a dataframe and store all the rmse and mae, and use the dataframe to get mean and std
rdd_err = sc.parallelize(error_list)
err_df = rdd_err.toDF(["rmse-1","rmse-2","rmse-3","ame-1","ame-2","ame-3"])

from pyspark.sql.functions import mean, stddev
# get each column's rmse and mae
error_df = err_df.select(mean('rmse-1').alias('mean-rmse-1'),stddev('rmse-1').alias('std-rmse-1'),\
                mean('rmse-2').alias('mean-rmse-2'),stddev('rmse-2').alias('std-rmse-2'),\
                mean('rmse-3').alias('mean-rmse-3'),stddev('rmse-3').alias('std-rmse-3'),\
                mean('ame-1').alias('mean-ame-1'),stddev('ame-1').alias('std-ame-1'),\
                mean('ame-2').alias('mean-ame-2'),stddev('ame-2').alias('std-ame-2'),\
                mean('ame-3').alias('mean-ame-3'),stddev('ame-3').alias('std-ame-3'))
# transform the dataframe of error_df into a python list of 'bridge_list'
bridge_list = error_df.collect()

# Visualise the mean and std of RMSE and MAE for each of the three versions of ALS in one single figure,
# I choose a dataframe to visualize them.
# a new dataframe's all the row data
final_data = []
final_data_plt = []
# rearrange the data and show all the rmse, mae, mean and std
for i in range(3):
    final_data.append((rmse_outter[i][0], rmse_outter[i][1], rmse_outter[i][2],mae_outter[i][0],mae_outter[i][1],\
                       mae_outter[i][2], bridge_list[0][i*2],bridge_list[0][i*2+1], bridge_list[0][6+i*2],\
                       bridge_list[0][7+i*2]))
    final_data_plt.append([rmse_outter[i][0], rmse_outter[i][1], rmse_outter[i][2],mae_outter[i][0],mae_outter[i][1],\
                       mae_outter[i][2], bridge_list[0][i*2],bridge_list[0][i*2+1], bridge_list[0][6+i*2],\
                       bridge_list[0][7+i*2]])

rdd_final = sc.parallelize(final_data)
# 'n-rmse' represents each fold's nth rmse, 'n-ame' represents each fold's nth ame
# 'rmse-mean' reresents each fold's mean of rmse, 'rmse-std reresents each fold's std of rmse
# 'ame-mean' reresents each fold's mean of ame, 'ame-std reresents each fold's std of ame
final_df = rdd_final.toDF(["1-rmse","2-rmse","3-rmse","1-ame","2-ame","3-ame","rmse-mean","rmse-std","ame-mean","ame-std"])
# show the dataframe with all the rmse, mae, mean and std
final_df.show()


# 2.C
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

# read genome-scores.csv, tags.csv and genome-tags.csv
df_genome = spark.read.load("../Data/genome-scores.csv", format="csv", inferSchema="true", header="true").cache()
df_tags = spark.read.load("../Data/tags.csv", format="csv", inferSchema="true", header="true").cache()
df_genome_tags = spark.read.load("../Data/genome-tags.csv", format="csv", inferSchema="true", header="true").cache()

# get the first 3 models which are corresponding three-fold 3 models
model_split = model_list[0:3]
# put all the data of tags and movie number into 'data_list"
data_list = []
for model in model_split:
    # use itemFactors to get 25 clusters
    dfItemFactorsCur = model.itemFactors
    factor_rdd = dfItemFactorsCur.rdd.map(lambda r: (r['id'],) + tuple([Vectors.dense(r['features'])]))
    dfItemFactors = factor_rdd.toDF(['id', 'features'])
    kmeans = KMeans(k=25, seed=190)
    model_k = kmeans.fit(dfItemFactors)
    # transform the model_k with 'id' and 'prediction', 'prediction' means the cluster id
    transformed=model_k.transform(dfItemFactors).select("id","prediction")
    # We use 'groupBy' to count each cluster's number and then sort them
    cluster_count=transformed.groupBy('prediction').count()
    cluster_sort=cluster_count.sort('count', ascending=False)
    # get the three largest clusters' predictions
    top3 = cluster_sort.collect()[:3]
    top3_type = [x[0] for x in top3]
    # get the three largest clusters by filter
    top3_cluster = []
    for i in top3_type:
        top3_cluster.append(transformed.filter(transformed.prediction == i))
    for cluster in top3_cluster:
        # to filter the df_genome with movieId in corresponding cluster's id
        cluster_genome = df_genome.join(cluster, df_genome.movieId == cluster.id)
        # use 'groupBy' to generate a new dataframe according their 'tagId' and add a column with their corresponding total scores
        score_cluster = cluster_genome.groupBy('tagId').sum('relevance')
        # sort the new dataframe according tags' total scores
        topTag =score_cluster.sort('sum(relevance)', ascending=False)
        # get the top three tags
        curTopTag  = topTag.collect()[0:3]
        # put the three tags into a list
        tags_list = [x[0] for x in curTopTag]
        # count the number of movies having each of these three tags
        for id in tags_list:
            # find the corresponding tag name in df_genome_tags
            tag_name = df_genome_tags.filter(df_genome_tags.tagId == id).collect()[0][1]
            # find a tag's all the rows in df_tags
            cur_tag_df = df_tags.filter(df_tags.tag == tag_name)
            # find all the movies in current cluster
            tag_with_movie_in_cluster = cur_tag_df.join(cluster, cur_tag_df.movieId == cluster.id)
            movie_counts = tag_with_movie_in_cluster.count()
            # print(movie_counts)
            data_list.append((id, tag_name, movie_counts))
print(data_list)
# choose a dataframe to visualise these data.
rdd = sc.parallelize(data_list)
data_df = rdd.toDF(["tag_id","tag_name","movies_number"])
data_df.show(27, False)







