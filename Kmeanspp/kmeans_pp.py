import pandas as pd
import numpy as np
import mykmeanssp
import sys
import math

# try:
#    #total arguments
#    arg_c = len(sys.argv)
#    print(sys.argv)
#    k = int(sys.argv[1])
#    iter_ = 300
#    epsilon = int(sys.argv[2])
#    file_name_1 = sys.argv[-2]
#    file_name_2 = sys.argv[-1]
#
#    if arg_c == 6:
#        iter_ = int(sys.argv[2])
#        epsilon = int(sys.argv[3])
#
# except:
#    print("An error has Occurred")
#    sys.exit()


# if arg_c > 4 or arg_c < 2:
#    print("An Error has Occurred")
#    sys.exit()

# if iter_ > 999:
#    print("Invalid maximum iteration!")
#    sys.exit()


file_name_1 = "input_1_db_1.txt"
file_name_2 = "input_1_db_2.txt"

k = 7
iter_ = 300
epsilon = 0.01

def euc_d(p1, p2, dim):
    sum_ = 0.0
    for i in range(dim):
        sum_ += (p1[i]-p2[i])**2

    return math.sqrt(sum_)

def init_centroids(data):
    np.random.seed(0)
    centroids = []

    prob = pd.Series(np.full((data.shape[0], 1), 1 / data.shape[0])[:, 0])

    first_centroid = data.iloc[np.random.choice(data.shape[0], p=prob)]
    centroids.append(first_centroid)

    for i in range(k - 1):  # init k-1 more centroids
        sum_dx = 0
        for j in range(data.shape[0]):  # calculate dx for each point
            p = data.iloc[j]
            dx = euc_d(centroids[len(centroids) - 1], p, len(p))  # first calculating distance to last centroid and then checking if there is a closer one.
            for m in range(len(centroids) - 1):
                dist_to_next_centroid = euc_d(centroids[m], p, len(p))
                dx = min(dx, dist_to_next_centroid)

            prob[j] = dx
            sum_dx += dx

        prob /= sum_dx
        new_centroid = data.iloc[np.random.choice(data.shape[0], p=prob)]
        centroids.append(new_centroid)

    return centroids



obs1 = pd.read_csv(file_name_1, header=None).values
obs2 = pd.read_csv(file_name_2, header=None).values
obs1_df = pd.DataFrame(obs1)
obs2_df = pd.DataFrame(obs2)

points = pd.merge(obs1_df, obs2_df, how='inner', on=obs1_df.columns[0], sort=True)
points_df = pd.DataFrame(points)
#srt_points = points_df.sort_values(points_df.columns[0]) already sorting in merge but they asked to sort after
points_df = points_df.iloc[:, 1:]
print(init_centroids(points_df))        #removing key
centroids = init_centroids(points_df)

# c function, takes data in a list form, and centroids in a list.
# mykmeanssp.fit(points.values.tolist(), centroids, iter_, epsilon)
