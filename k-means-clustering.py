'''
Rhea Kapur
k-Means Clustering
Image Segmentation
1/26/2020
'''

import numpy as np
from numpy.linalg import norm
import pandas as pd
import random
from PIL import Image
import math

np.seterr('raise')

img = Image.open("YOUR_IMAGE_FILEPATH")
A = np.asarray(img, dtype=np.float32)
arr = A[:,:,:3] # Remove the alpha channel, not segmenting with it

img.show() # original

nbRows = A.shape[0]
nbCols = A.shape[1]
color = A.shape[2]
nbPixels = nbRows * nbCols

k = 4

# stored as indices
# stored as indices
rrvals = random.sample(range(nbRows), k)
crvals = random.sample(range(nbCols), k)

centroids = []
for x in range(0,k):
    centroids.append(arr[rrvals[x]][crvals[x]])

max_iter = 20

# new image
new_arr = np.zeros(shape=arr.shape)

# distance between 2 pixels
def compute_distance(p0, p1):
    return math.sqrt(float((p0[0] - p1[0])**2) + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)

def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

for a in range(0, max_iter):
    print("Iteration: " + str(a))

    cluster_num = np.zeros(k)
    cluster_tval = np.zeros(shape=(k,3)) # r, g, b as cols, clusters 1-k as rows

    # x pixel
    for x in range(0, nbRows):
        # y pixel
        for y in range(0, nbCols):
                min_dist = float('inf')
                correct_centroid = -1

                for c_idx, centroid in enumerate(centroids):
                    current_dist = compute_distance(centroid, arr[x][y])

                    if(current_dist < min_dist):
                        min_dist = current_dist
                        correct_centroid = c_idx

                new_arr[x][y] = centroids[correct_centroid]

                cluster_num[correct_centroid] += 1 # incrementing count of how many in the centroid

                # gets r, g, b vals from original point (unchanged), that which has now been assigned color of cluster center
                for z in range(0, 3):
                    cluster_tval[correct_centroid][z] += arr[x][y][z]

    # centroid
    for c_idx in range(0,k):
        # rgb
        for b in range(0,3):
            centroids[c_idx][b] = round(cluster_tval[c_idx][b]/float(cluster_num[c_idx]))

Image.fromarray(new_arr.astype(np.uint8)).show()
