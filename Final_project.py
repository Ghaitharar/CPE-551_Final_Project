
import twint
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from Historic_Crypto import HistoricalData


date_since = input("Enter your the start date of analysis: (YYYY-MM-DD) \n")
print("You entered: \n", date_since)


pd.set_option('display.max_columns', None)
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
plt.close("all")


def scrap_ETH_tweets(start_date):
    #Scrap tweets that contain "Ethereum" or "ETH" begining from the specified date. The resoult saved in csv file on disk.
    c = twint.config.Config()
    c.Search = "Ethereum", "ETH"
    c.Since = start_date
    c.Store_csv = True
    c.Output = "Test_Output.csv"
    c.Count = True
    c.Stats = False
    c.Hide_output = True
    twint.run.Search(c)

def load_scrapped_tweets(tweetfile):
    Tweets_data = pd.read_csv(tweetfile, skipinitialspace=True, usecols=['date','tweet'])
    print("csv file loaded")
    Tweets_data['date'] = pd.to_datetime(Tweets_data['date']) #convert date to standard format
    Tweets_data['neg'] = 0.0    #Add column for negative score
    Tweets_data['neu'] = 0.0    #Add column for neutral score
    Tweets_data['pos'] = 0.0    #Add column for positive score
    return Tweets_data


def get_score(tweets_dataframe):
    print("Getting the score of each tweet")
    for index_T, row_T in tweets_dataframe.iterrows():
        #print(index_T)
        temp_score = sia.polarity_scores(row_T['tweet'])
        tweets_dataframe.at[index_T, "neg"] = temp_score['neg']
        tweets_dataframe.at[index_T, "neu"] = temp_score['neu']
        tweets_dataframe.at[index_T, "pos"] = temp_score['pos']
    return tweets_dataframe

def dateframe_aggregate(scores_dataframe):
    print("aggregation_functions")
    aggregation_functions = {'neg': 'mean', 'neu': 'mean', 'pos': 'mean'}
    brief_tweets_data = scores_dataframe.groupby(scores_dataframe['date']).aggregate(aggregation_functions)
    print("Done!")
    return brief_tweets_data

def plot_scores(Plote_dataframe):
    print(Plote_dataframe)
    #plt.figure()
    #Plote_dataframe.plot()
    #plt.show()

def get_ETH_data(granularity=86400, date='2021-05-17-00-00'):
    ETH_dataframe = HistoricalData('ETH-USD',granularity,date).retrieve_data()
    #new.to_csv(r'C:/Users/ghait/PycharmProjects/CryptoPred/export_dataframe.csv',index=True)
    return ETH_dataframe


print("Getting Tweets...")
scrap_ETH_tweets(date_since)
tweets_data = load_scrapped_tweets("Test_Output.csv")
print("Getting Tweets Done!")

print("Getting scores...")
scored_tweets_data = get_score(tweets_data)
print("Getting scores Done!")

print("Aggregating data...")
aggregated_data = dateframe_aggregate(scored_tweets_data)
plot_scores(aggregated_data)
ETH_date = get_ETH_data(date=date_since+'-00-00')

result = pd.concat([aggregated_data, ETH_date], axis=1)
print(result)
print(result.corr())
