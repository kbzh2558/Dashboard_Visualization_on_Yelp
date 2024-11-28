import pandas as pd

# load data
attribute = pd.read_csv('data_processed\\filtered_yelp_business_attributes.csv')
business = pd.read_csv('data_processed\\filtered_yelp_business.csv')    
checkin = pd.read_csv('data_processed\\filtered_yelp_checkin.csv')
review = pd.read_csv('data_processed\\filtered_yelp_review.csv')    
tip = pd.read_csv('data_processed\\filtered_yelp_tip.csv')

# encode attribute cols
# mapping
value_map = {'False': 1, 'Na': 0, 'True': 2}

# apply the mapping to the entire df
attribute = attribute.map(lambda x: value_map.get(x, x))

# merge df
# left joins on `business_id`
# Load and select specific columns from business
merged_df = business[['business_id', 'stars', 'review_count']]

# merge attribute DataFrame with a prefix
merged_df = merged_df.merge(
    attribute.add_prefix('attr_'), 
    left_on='business_id', 
    right_on='attr_business_id', 
    how='left'
).drop(columns=['attr_business_id'])

merged_df = merged_df.merge(
    checkin.groupby('business_id').agg({'checkins': 'sum'}).rename(columns={'checkins': 'checkin_total'}), 
    on='business_id', 
    how='left'
)

merged_df = merged_df.merge(
    review.groupby('business_id').agg({'sentiment': 'mean', 'subjectivity': 'mean'}).rename(
        columns={'sentiment': 'review_sentiment_mean', 'subjectivity': 'review_subjectivity_mean'}
    ),
    on='business_id', 
    how='left'
)

merged_df = merged_df.merge(
    tip.groupby('business_id').agg({'sentiment': 'mean', 'subjectivity': 'mean'}).rename(
        columns={'sentiment': 'tip_sentiment_mean', 'subjectivity': 'tip_subjectivity_mean'}
    ),
    on='business_id', 
    how='left'
)

# fill na
# define lists of col names
attributes_cols = [col for col in merged_df.columns if col.startswith('attr_')]
sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col]
subjectivity_cols = [col for col in merged_df.columns if 'subjectivity' in col]
checkin_cols = ['checkin_total']  

# Fill missing values according to the specified rules
merged_df[attributes_cols] = merged_df[attributes_cols].fillna(0)
merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
merged_df[subjectivity_cols] = merged_df[subjectivity_cols].fillna(0.5)
merged_df[checkin_cols] = merged_df[checkin_cols].fillna(0)

# save csv
merged_df.to_csv('data_processed\\yelp_clusterinput_processed.csv', index=False)



