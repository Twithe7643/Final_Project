import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx

#Loads Data
cities_df = pd.read_csv(r"C:\Users\Twith\OneDrive\414INST Work\Module 3\us-cities-demographics.csv",
sep=';')
print(cities_df)
print(cities_df.shape)
print(cities_df.columns)
print(cities_df.head())

#Columns chosen 
city_features = ['Median Age',
                 'Total Population',
                 'Number of Veterans',
                 'Female Population',
                 'Male Population',
                 'Average Household Size'
]

city_feature_df = cities_df.groupby(['City', 'State'])[city_features].mean(numeric_only=True).reset_index()
print(city_feature_df)


city_df_clean = city_feature_df.dropna(subset=city_features) # Drop rows with missing numeric features
print(city_df_clean[city_features].isna().sum()) # Confirm no missing values remain

city_df_clean['City_State'] = city_df_clean['City'] + ", " + city_df_clean['State']

# Scale and compute similarity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(city_df_clean[city_features])

similarity = cosine_similarity(X_scaled) #Computes cosine similarity
similarity_df = pd.DataFrame(similarity, index=city_df_clean['City_State'], columns=city_df_clean['City_State'])
print(similarity_df) #Prints Dataframe with similarity values

query_cities = ['Chicago, Illinois', 'Cary, North Carolina', 'New York, New York', 'Anaheim, California'] #Chosen Cities

for city in query_cities:
    top_10 = similarity_df[city].sort_values(ascending=False)[1:11]  #Skips self
    print(f"\nTop 10 cities similar to {city}:")
    print(top_10)


for city in query_cities:
    top_10 = similarity_df[city].sort_values(ascending=False)[1:11]  #Skips Self
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_10.values, y=top_10.index, palette="viridis") #Barchart
    plt.title(f"Top 10 Cities Similar to {city}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("City, State")
    plt.xlim(0,1)  # Cosine similarity ranges from 0 to 1
    plt.tight_layout()
    plt.show()

#Clustering Analysis
sil_score = [] 
k_range = range(2,10) #Range for the model
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_score.append(silhouette_score(X_scaled, labels)) #Measures city fit

best_index = sil_score.index(max(sil_score))
best_k = list(k_range)[best_index] #Chooses best K from range

final_model = KMeans(n_clusters=best_k, random_state=42) 
city_df_clean['Cluster'] = final_model.fit_predict(X_scaled) #Assigns each city to a demographic cluster
cluster_summary = city_df_clean.groupby('Cluster')[city_features].mean()
print("\nCluster summary (average demographics):")
print(cluster_summary)

cluster_cities = city_df_clean.groupby('Cluster')['City_State'].apply(list) #Lists which cities belong to each cluster

for cluster, cities in cluster_cities.items():
    print(f"\nCluster {cluster} cities ({len(cities)} total):")
    print(cities[:10], "...") 


plt.figure(figsize=(8,5)) #Plots line graph
plt.plot(list(k_range), sil_score, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for K-Means Clustering")
plt.tight_layout()
plt.show()


#Network Analysis
# Network Analysis: City Similarity Network (Top-N Neighbors, Cluster-Colored)

top_n = 5
G = nx.Graph() 

# Build network using top-N cosine similarities for selected cities
for city in query_cities:
    top_similar = similarity_df[city].sort_values(ascending=False)[1:top_n+1]
    for neighbor, sim_value in top_similar.items():
        G.add_edge(city, neighbor, weight=sim_value)#Adds edges between cities

# Subgraph: query cities + their neighbors
subset_nodes = set(query_cities)
for city in query_cities:
    subset_nodes.update(G.neighbors(city))

subgraph = G.subgraph(subset_nodes)

# Map each city to its cluster
city_to_cluster = dict(
    zip(city_df_clean['City_State'], city_df_clean['Cluster'])
)

# Color nodes by cluster
node_colors = [city_to_cluster[node] for node in subgraph.nodes()]#Colors nodes by cluster
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(subgraph, seed=42, k=0.8)
nx.draw_networkx_nodes(
    subgraph,
    pos,
    node_color=node_colors,
    cmap=plt.cm.tab10,
    node_size=500
)
nx.draw_networkx_edges(
    subgraph,
    pos,
    width=[d['weight'] * 3 for (_, _, d) in subgraph.edges(data=True)],
    alpha=0.6
)
nx.draw_networkx_labels(subgraph, pos, font_size=9)
plt.title(
    "City Similarity Network (Top 5 Demographic Neighbors, Colored by Cluster)",
    fontsize=16
)
plt.axis("off")
plt.show()
