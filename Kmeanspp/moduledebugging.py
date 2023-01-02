import mykmeanssp
import numpy
import pandas as pd

obs1 = pd.read_csv("input_2.txt", header=None)
points_df = obs1.values.tolist()
smaller_df = points_df[:7]


mykmeanssp.fit(points_df, smaller_df, 200, 0.01)
