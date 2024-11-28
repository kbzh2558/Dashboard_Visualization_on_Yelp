import pandas as pd

df = pd.read_csv('data_processed\\filtered_yelp_business.csv')

# cat = []
# for i in df.categories.unique():
#     for j in i.split(';'):
#         if j not in cat:
#             cat.append(j)

# Define the main categories and their representative labels
category_mapping = {
    "Food & Dining": ["Restaurants", "Food", "Coffee & Tea",'Bars','Specialty Food'],
    "Health & Medical": ["Health & Medical",'Doctors','Eyewear & Opticians','Hospitals'],
    "Home Services": ["Home Services",'Home Decor','Pet Services','Pest Control','Air Duct Cleaning','Plumbing'],
    "Shopping": ["Shopping",'Jewelry','Accessories','Fashion','Grocery'],
    "Entertainment": ["Arts & Entertainment",'Art Schools','Party & Event Planning','Karaoke'],
    "Professional Services": ["Financial Services",'Automotive','IT Services & Computer Repair','Appliances & Repair'],
    "Fitness & Recreation": ["Active Life", "Gyms", "Fitness & Instruction",'Boxing','Golf']
}

# Function to categorize each row
def map_category(row):
    categories = row.split(';')
    for main_category, keywords in category_mapping.items():
        if any(keyword in categories for keyword in keywords):
            return main_category
    return 'Other'

# Apply the function to the DataFrame
df['main_category'] = df['categories'].apply(map_category)

df.to_csv('data_processed\\filtered_yelp_business_cate.csv', index=False)
