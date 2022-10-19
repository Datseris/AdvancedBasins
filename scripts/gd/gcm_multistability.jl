using DrWatson
@quickactivate
using DynamicalSystems
using DelimitedFiles

data = readdlm(datadir("gcm_multistability", "data.csv"), ';')
# data have header and numbers rows
names = string.(data[1, :])
all_features = [float.(data[j, :]) for j in 2:size(data, 1)]

# Charline Ragon suggested to use the following "features":
# water mass transport peak, precipitations, components of the LEC,
# contributions from MEP, temperature gradient and heat transports
suggested_feature_idxs = [7, 9, 10, 14, 23]
reduced_features = [f[suggested_feature_idxs] for f in all_features]

# Let's see what our clustering will do..., will we find one cluster for each feature?
cconfig = ClusteringConfig(; min_neighbors = 1)

cluster_labels, cluster_errors = cluster_features(reduced_features, cconfig)