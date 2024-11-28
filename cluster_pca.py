import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# cluster
from sklearn.cluster import KMeans
# pca
from sklearn.decomposition import PCA



# load data
df = pd.read_csv('data_processed\\yelp_clusterinput_processed.csv', low_memory=False)

# drop cols with uneeded cols
object_cols = df.drop(columns='business_id').select_dtypes(include=['object']).columns
df = df.drop(columns=object_cols)

# drop cols with unary data
df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])

# drop id
data = df.drop(columns='business_id')


# Group 1: Columns starting with "attr"
attr_columns = [col for col in data.columns if col.startswith("attr")]

# Group 2: Columns containing "sentiment" or "subjectivity"
sentiment_subjectivity_columns = [col for col in data.columns if "sentiment" in col or "subjectivity" in col]

# Group 3: The remaining columns (excluding those we need to ignore and the ones above)
other_columns = [
    col for col in data.columns 
    if col not in attr_columns and col not in sentiment_subjectivity_columns
]

user_reaction_columns = sentiment_subjectivity_columns + other_columns

# separate df for each group
attr_data = data[attr_columns]
user_reaction_data = data[user_reaction_columns]

# pca with 1 component for each group
pca_attr = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(attr_data))
pca_user_reaction = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(user_reaction_data))



# save to output
df['PCA_Attr'] = pca_attr.flatten()
df['PCA_User_Reaction'] = pca_user_reaction.flatten()


from sklearn.cluster import AgglomerativeClustering

# Combine two PC1 components
data = np.concatenate((pca_attr.flatten().reshape(-1, 1), pca_user_reaction.flatten().reshape(-1, 1)), axis=1)

# Normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Hierarchical (Agglomerative) clustering
hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(normalized_data)



output = pd.merge(
    df,
    pd.DataFrame(list(zip(df['business_id'],np.transpose(hierarchical_labels))), columns = ['business_id','cluster_label']),
    on='business_id'
)


# # Combine all labels into a DataFrame
# output = pd.DataFrame({
#     'business_id': df['business_id'],
#     'hierarchical_cluster': hierarchical_labels,
# })

# Save to CSV
output.to_csv('data_processed\\yelp_cluster_pca.csv', index=False)







