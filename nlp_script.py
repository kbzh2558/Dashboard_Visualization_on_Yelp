
# package
import pandas as pd
from pandarallel import pandarallel

# initialize
pandarallel.initialize()

# def func
def get_sentiment(text: str)->float:
    """Get sentiment score from text.
    The function uses TextBlob to calculate the sentiment score of the text if the text is not empty.
    
    Args:
        text (str): text to analyze

    Returns:
        sentiment score (float)
    """
    from textblob import TextBlob
    text = str(text) if not isinstance(text, str) else text
    if len(text) != 0:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    else: 
        return 0
    
def get_subjectivity(text: str)->float:
    """Get subjectivity score from text
    The function uses TextBlob to calculate the subjectivity score of the text if the text is not empty.

    Args:
        text (str): text to analyze

    Returns:
        subjectivity score (float)    
    """
    from textblob import TextBlob
    
    text = str(text) if not isinstance(text, str) else text
    if len(text) != 0:
        blob = TextBlob(text)
        return blob.sentiment.subjectivity
    else:
        return 0.5

def main():
    review = pd.read_csv('data\\yelp_review_subsample.csv')    
    tip = pd.read_csv('data\\yelp_tip.csv')
    # attribute = pd.read_csv('data\yelp_business_user.csv')    


    # add sentiment, subjectivity col
    review.loc[:,'sentiment'] = review['text'].parallel_apply(get_sentiment)
    review.loc[:,'subjectivity'] = review['text'].parallel_apply(get_subjectivity)
    tip.loc[:,'sentiment'] = tip['text'].parallel_apply(get_sentiment)
    tip.loc[:,'subjectivity'] = tip['text'].parallel_apply(get_subjectivity)

    # save files
    review.to_csv('data\\yelp_business_review_subsample_processed.csv', index=False)
    tip.to_csv('data\\yelp_business_tip_processed.csv', index=False)

# aggregation
# checkin.groupby('business_id').agg({'checkins':'sum'}),

if __name__ == "__main__":
    main()

