import pandas as pd

# Load data
attribute = pd.read_csv('data\\yelp_business_attributes.csv')
business = pd.read_csv('data\\yelp_business.csv')
hours = pd.read_csv('data\\yelp_business_hours.csv')
checkin = pd.read_csv('data\\yelp_checkin.csv')
review = pd.read_csv('data\\yelp_business_review_subsample_processed.csv')
tip = pd.read_csv('data\\yelp_business_tip_processed.csv')
user = pd.read_csv('data\\yelp_user.csv')


# Step 1: Sample unique `business_id` and `user_id` from business and users data (assuming `review` and `tip` have user information)
sampled_business_ids = business['business_id'].sample(6000, replace=False, random_state=42).unique()

sampled_user_ids = review['user_id'].sample(100000, replace=False, random_state=42).unique()

# Step 2: Filter each DataFrame to only keep rows with the sampled `business_id`s and `user_id`s

# Filter business-related tables by `business_id`
business_filtered = business[business['business_id'].isin(sampled_business_ids)]
user_filtered = user[user['user_id'].isin(sampled_user_ids)]
attribute_filtered = attribute[attribute['business_id'].isin(sampled_business_ids)]
checkin_filtered = checkin[checkin['business_id'].isin(sampled_business_ids)]
review_filtered = review[(review['business_id'].isin(sampled_business_ids)) & (review['user_id'].isin(sampled_user_ids))]
tip_filtered = tip[(tip['business_id'].isin(sampled_business_ids)) & (tip['user_id'].isin(sampled_user_ids))]
hours_filtered = hours[hours['business_id'].isin(sampled_business_ids)]


# Save or use the filtered data
business_filtered.to_csv('data_processed\\filtered_yelp_business.csv', index=False)
attribute_filtered.to_csv('data_processed\\filtered_yelp_business_attributes.csv', index=False)
hours_filtered.to_csv('data_processed\\filtered_yelp_business_hours.csv', index=False)
checkin_filtered.to_csv('data_processed\\filtered_yelp_checkin.csv', index=False)
review_filtered.to_csv('data_processed\\filtered_yelp_review.csv', index=False)
tip_filtered.to_csv('data_processed\\filtered_yelp_tip.csv', index=False)
user_filtered.to_csv('data_processed\\filtered_yelp_user.csv', index=False)

