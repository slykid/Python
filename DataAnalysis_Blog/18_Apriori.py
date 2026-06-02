import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from apyori import apriori

data = pd.read_csv("Dataset/groceries.csv", sep=",")