from sklearn import datasets
import numpy as np


# general config
n_samples = 500 #280
noise = 0 # .05

# circles
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=noise)

# moons
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)

# blobs
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

# no structure
no_structure = np.random.rand(n_samples, 2), None

# anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

# save all the generated data sets together
datasets = [
    noisy_circles,
    noisy_moons,
    varied,
    aniso,
    blobs,
    no_structure]