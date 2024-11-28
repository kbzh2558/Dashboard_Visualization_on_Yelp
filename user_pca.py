import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# cluster
from sklearn.cluster import KMeans
# pca
from sklearn.decomposition import PCA



# load data
df = pd.read_csv('data_processed\\yelp_user_clusterinput_processed.csv', low_memory=False)

# drop cols with uneeded cols
object_cols = df.drop(columns='user_id').select_dtypes(include=['object']).columns
df = df.drop(columns=object_cols)

# drop cols with unary data
df = df.drop(columns=[col for col in df.columns if df[col].nunique() == 1])

# drop id
data = df.drop(columns='user_id')


# Group columns
user_profile_columns = [
    'friends_count', 'elite_length', 'average_stars', 
    'days_yelping', 'compliment_profile', 'compliment_cute', 
    'compliment_list', 'compliment_note', 'compliment_plain', 
    'compliment_photos'
]

user_activity_columns = [
    'review_count', 'useful', 'funny', 'cool', 'fans', 
    'compliment_hot', 'compliment_more', 'compliment_cool', 
    'compliment_funny', 'compliment_writer', 'review_sentiment_mean', 
    'review_subjectivity_mean', 'tip_sentiment_mean', 'tip_subjectivity_mean'
]

# Extract data for PCA
user_profile_data = data[user_profile_columns].select_dtypes(include=[float, int])
user_activity_data = data[user_activity_columns].select_dtypes(include=[float, int])


# pca with 1 component for each group
pca_user_profile = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(user_profile_data))
pca_user_activity = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(user_activity_data))

# # save to output
df['PCA_User_Profile'] = pca_user_profile.flatten()
df['PCA_User_Activity'] = pca_user_activity.flatten()

data = np.concatenate((pca_user_profile.flatten().reshape(-1, 1), pca_user_activity.flatten().reshape(-1, 1)), axis=1)


# normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# kmeans
kmeans1 = KMeans(n_clusters=3, n_init='auto',random_state=128)
model = kmeans1.fit(normalized_data)
labels = model.predict(normalized_data)

# join labels to df
output = pd.merge(
    df,
    pd.DataFrame(list(zip(df['user_id'],np.transpose(labels))), columns = ['user_id','cluster_label']),
    on='user_id'
)

# save csv
output.to_csv('data_processed\\yelp_user_cluster_pca.csv', index=False)





