import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# dataset
np.random.seed(42)
segment1 = np.random.normal(loc=30, scale=5, size=100)
segment2 = np.random.normal(loc=60, scale=10, size=150)
segment3 = np.random.normal(loc=90, scale=8, size=120)
data = np.concatenate([segment1,segment2,segment3]).reshape(-1,1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

n_components = 3
gmm = GaussianMixture(n_components = n_components, random_state = 42)
gmm.fit(data_scaled)

# prediction
cluster_labels = gmm.predict(data_scaled)

# visualize
plt.scatter(data,np.zeros_like(data), c=cluster_labels, cmap='viridis')
plt.title('Cluster Segmentation')
plt.xlabel('Purchase Amount')
plt.show()

# user input
user_input = float(input("Enter a purchase amount to predict the coustomer segment: "))
user_input_scaled = scaler.transform(np.array([[user_input]]))
prediceted_label=gmm.predict(user_input_scaled.reshape(-1,1))[0]
print(f"Th predicted customer segmnt for a purchase amount of {user_input} is : {prediceted_label + 1}")

