import pandas as pd
from datetime import datetime

# load data
review = pd.read_csv('data_processed\\filtered_yelp_review.csv')    
tip = pd.read_csv('data_processed\\filtered_yelp_tip.csv')
user = pd.read_csv('data_processed\\filtered_yelp_user.csv')




user['yelping_since'] = pd.to_datetime(user['yelping_since'], errors='coerce')
# Calculate the difference in days from today
user['days_yelping'] = (datetime.now() - user['yelping_since']).dt.days

# Step 2: Calculate the number of friends
# Split the 'friends' column by comma and count the elements
user['friends_count'] = user['friends'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
user['elite_length'] = user['elite'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

# Step 3: Fill all NaN values with 0
user.fillna(0, inplace=True)


# merge df
merged_df = user.drop(columns=['yelping_since'])

merged_df = merged_df.merge(
    review.groupby('user_id').agg({'sentiment': 'mean', 'subjectivity': 'mean'}).rename(
        columns={'sentiment': 'review_sentiment_mean', 'subjectivity': 'review_subjectivity_mean'}
    ),
    on='user_id', 
    how='left'
)

merged_df = merged_df.merge(
    tip.groupby('user_id').agg({'sentiment': 'mean', 'subjectivity': 'mean'}).rename(
        columns={'sentiment': 'tip_sentiment_mean', 'subjectivity': 'tip_subjectivity_mean'}
    ),
    on='user_id', 
    how='left'
)

# fill na
# define lists of col names
sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col]
subjectivity_cols = [col for col in merged_df.columns if 'subjectivity' in col]


# Fill missing values according to the specified rules
merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
merged_df[subjectivity_cols] = merged_df[subjectivity_cols].fillna(0.5)

# save csv
merged_df.to_csv('data_processed\\yelp_user_clusterinput_processed.csv', index=False)