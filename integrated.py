"""Maximum number of reviews in a day
Percentage of reviews with positive/negative ratings
Average review length
The standard deviation of ratings of the reviewer’s reviews
**Review-Text features**


# Dependencies
"""
import pandas as pd
# Imports
import streamlit as st
from bertopic import BERTopic
from textblob import TextBlob
from umap import UMAP
from hdbscan import HDBSCAN
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from langdetect import detect
import sklearn
print(sklearn.__version__)
import contractions
#Scraper Imports
import requests
from bs4 import BeautifulSoup
import re, sys

# function to plot most frequent terms
import nltk
nltk.download("popular")
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
stop_words = set(stopwords.words('english'))
import re

# Code for project structure
st.image('piovos_logo.png')
st.title("Piovis Automate")
st.sidebar.title('Review analyzer GUI')
st.markdown("This application is a streamlit deployment to automate analysis")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df)
else:
    st.stop()
data= df
#functions

# Making result human friendly
def get_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Loading Data


# Applying language detection
df.dropna(inplace=True)

text_col = df['review-text'].astype(str)
langdet = []

# Data preprocessing
for i in range(len(df)):
    try:
        lang=detect(text_col[i])
    except:
        lang='no'

    langdet.append(lang)

df['detect'] = langdet

# Select language here
en_df = df[df['detect'] == 'en']




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import sklearn
print(sklearn.__version__)

#Scraper Imports
import requests
from bs4 import BeautifulSoup
import re, sys

#Pre-Processing Imports
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import contractions

#Feature Extraction Imports

#Sentiment
from textblob import TextBlob

#POS Tagging
import nltk
nltk.download("popular")
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
stop_words = set(stopwords.words('english'))
import re

#Text Feature Generation
import string

#Model Training and Evaluation Imports
from time import time
from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import joblib
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold

POS = ['Noun_Count', 'Adj_Count', 'Verb_Count', 'Adv_Count', 'Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count', 'Nega_Count', 'Aux_Count']
array_Noun=[]
array_Adj=[]
array_Verb=[]
array_Adv=[]
array_Pro=[]
array_Con=[]
array_Art=[]
array_Nega=[]
array_Pre=[]
array_Aux=[]
Values = [array_Noun, array_Adj, array_Verb, array_Adv, array_Pro, array_Pre, array_Con, array_Art, array_Nega, array_Aux]
i = 0
for x in POS:
    data[x] = pd.Series(Values[i])
    data[x] = data[x].fillna(0)
    data[x] = data[x].astype(float)
    i += 1





"""# **Final Dataset**"""

import datefinder
data = en_df
#content= data['review-date'].astype('str').apply(datefinder.find_dates(data['review-date']))
#content

#content

"""**Removing Emojis from text**"""

data['review-text'] = data['review-text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

data['review-text'][0]

"""**Fix Contractions**"""

def fixContra(text):
    return contractions.fix(text)

data['review-text'] = data['review-text'].apply(lambda x: fixContra(x))

"""**Remove Special Character**"""

# \W represents Special characters
data['review-text'] = data['review-text'].str.replace('\W', ' ')


# \d represents Numeric digits
data['review-text'] = data['review-text'].str.replace('\d', ' ')

"""**Upper to Lower Case**"""

data['review-text'] = data['review-text'].str.lower()


data.head()



"""# **Sentiment Score Generation**"""

reviews = data['review-text'].tolist()
#print(reviews)
sentiment_score = []
sentiment_subjectivity=[]
review_head_sentiment=[]
for rev in reviews:
    testimonial = TextBlob(rev)
    sentiment_score.append(testimonial.sentiment.polarity)
    sentiment_subjectivity.append(testimonial.sentiment.subjectivity)

data['Sentiment'] = sentiment_score
data['Subjectivity'] = sentiment_subjectivity
data.head()

"""**Visualizing the sentiment**"""

pos = 0
neg = 0
for score in data['Sentiment']:
    if score > 0:
        pos += 1
    elif score < 0:
        neg += 1

#Visualiing the distribution of Sentiment
values = [pos, neg]
label = ['Positive Reviews', 'Negative Reviews']

fig = plt.figure(figsize =(10, 7))
plt.pie(values, labels = label)

plt.show()

#Number of Negative words in a review
reviews = data['review-text'].tolist()
negative_count = []
for rev in reviews:
    words = rev.split()
    neg = 0
    for w in words:
        testimonial = TextBlob(w)
        score = testimonial.sentiment.polarity
        if score < 0:
            neg += 1
    negative_count.append(neg)

data['Neg_Count'] = negative_count

"""# **Unique words count**"""



#Word Count
data['Word_Count'] = data['review-text'].str.split().str.len()

for i in range(data.shape[0]):
    if data.loc[i].Word_Count == 0:
        data.drop(index=i, inplace=True)
data.reset_index(drop=True, inplace=True)

reviews = data['review-text'].str.lower().str.split()

# Get amount of unique words
data['Unique_words'] = reviews.apply(set).apply(len)
#data['Unique_words'] = data[['Unique_words']].div(data.Word_Count, axis=0)

data

"""# **POS - Tagging**"""

import re
review_text = data['review-text']

array_Noun = []
array_Adj = []
array_Verb = []
array_Adv = []
array_Pro = []
array_Pre = []
array_Con = []
array_Art = []
array_Nega = []
array_Aux = []

articles = ['a', 'an', 'the']
negations = ['no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'barely', 'scarcely']
auxilliary = ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'do', 'does', 'did', 'have', 'having', 'has', 'had']

for j in review_text:
    text = j ;
    filter=re.sub('[^\w\s]', '', text)
    conver_lower=filter.lower()
    Tinput = conver_lower.split(" ")

    for i in range(0, len(Tinput)):
        Tinput[i] = "".join(Tinput[i])
    UniqW = Counter(Tinput)
    s = " ".join(UniqW.keys())

    tokenized = sent_tokenize(s)

    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        #wordsList = [w for w in wordsList if not w in stop_words]

        Art = 0
        Nega = 0
        Aux = 0
        for word in wordsList:
            if word in articles:
                Art += 1
            elif word in negations:
                Nega += 1
            elif word in auxilliary:
                Aux += 1

        tagged = nltk.pos_tag(wordsList)
        counts = Counter(tag for word,tag in tagged)

        N = sum([counts[i] for i in counts.keys() if 'NN' in i])
        Adj = sum([counts[i] for i in counts.keys() if 'JJ' in i])
        Verb = sum([counts[i] for i in counts.keys() if 'VB' in i])
        Adv = sum([counts[i] for i in counts.keys() if 'RB' in i])
        Pro = sum([counts[i] for i in counts.keys() if (('PRP' in i) or ('PRP$' in i) or ('WP' in i) or ('WP$' in i))])
        Pre = sum([counts[i] for i in counts.keys() if 'IN' in i])
        Con = sum([counts[i] for i in counts.keys() if 'CC' in i])

        array_Noun.append(N)
        array_Adj.append(Adj)
        array_Verb.append(Verb)
        array_Adv.append(Adv)
        array_Pro.append(Pro)
        array_Pre.append(Pre)
        array_Con.append(Con)
        array_Art.append(Art)
        array_Nega.append(Nega)
        array_Aux.append(Aux)
print('Completed')

POS = ['Noun_Count', 'Adj_Count', 'Verb_Count', 'Adv_Count', 'Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count', 'Nega_Count', 'Aux_Count']
Values = [array_Noun, array_Adj, array_Verb, array_Adv, array_Pro, array_Pre, array_Con, array_Art, array_Nega, array_Aux]
i = 0
for x in POS:
    data[x] = pd.Series(Values[i])
    data[x] = data[x].fillna(0)
    data[x] = data[x].astype(float)
    i += 1

"""# **Authenticity**"""

data

data = data.assign(Authenticity = lambda x: (x.Pro_Count + x.Unique_words - x.Nega_Count) / x.Word_Count)

"""# **Analytical Thinking**"""

data = data.assign(AT = lambda x: 30 + (x.Art_Count + x.Pre_Count - x.Pro_Count - x.Aux_Count - x.Con_Count - x.Adv_Count - x.Nega_Count))

"""# **Labelling the Reviews**"""

data.to_csv('before_labeling.csv')



def label(Auth, At, N, Adj, V, Av, S, Sub, W):
    score = 0
    if Auth >= 0.49:
        score += 2
    if At <= 20:
        score += 1
    if (N + Adj) >= (V + Av):
        score += 1
    if -0.5 <= S <= 0.5:
        score += 1
    if Sub <= 0.5:
        score += 2
    if W > 75:
        score += 3
    if score >= 5:
        return 1
    else:
        return 0

data['Rev_Type'] = data.apply(lambda x: label(x['Authenticity'], x['AT'], x['Noun_Count'], x['Adj_Count'], x['Verb_Count'], x['Adv_Count'], x['Sentiment'], x['Subjectivity'], x['Word_Count']), axis = 1)

data['Rev_Type'].value_counts()

data.head()

"""# preprocessing"""





# Removing text for transformation
data['rating-count'] = data['rating-count'].astype('category')
data['rating-avg'] = data['rating-avg'].astype('category')
data['review-text'] = data['review-text'].astype('category')

data.groupby(['asin', 'Review Score'])['Review Score'].count()

data['rating-count']= data['rating-count'].str.strip(' global ratings')
data['rating-avg']= data['rating-avg'].str.strip(' out of 5')

import nums_from_string
# Extracting nums from textual representation
# Refactor into function
data['rating-count'] = data['rating-count'].astype(str).apply(nums_from_string.get_nums)
data['rating-avg'] = data['rating-avg'].astype(str).apply(nums_from_string.get_nums)

# converting num lists into actual float
data['rating-count']= data['rating-count'].apply(pd.to_numeric, errors='coerce').astype(float)
data['rating-avg']= data['rating-avg'].apply(pd.to_numeric, errors='coerce').astype(float)

data

"""# **Model Training**"""

df = data.loc[:, data.columns[4:-1]]
df.drop(['review-text','Neg_Count','Unique_words','Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count',
       'Nega_Count', 'Aux_Count','review-rating','review-pagination','review-date','review-author','detect'], axis=1, inplace=True)

df

min_max_scaler = preprocessing.MinMaxScaler()
Columns=df.columns
df[Columns] = min_max_scaler.fit_transform(df[Columns])

df
en_df['human_sentiment'] = en_df['Sentiment'].apply(get_analysis)
bad_reviews = en_df[en_df['human_sentiment'] == 'Negative']
good_reviews = en_df[en_df['human_sentiment'] == 'Positive']
st.header('Select Stop Words')

custom_stopwords = st.text_input('Enter Stopword')
custom_stopwords = custom_stopwords.split()
nltk_Stop= stopwords.words("english")
final_stop_words = nltk_Stop + custom_stopwords
def clean_text(dataframe, col_name):

    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    word = "inversely"
    print("stemming:", stem.stem(word))
    print("lemmatization:", lem.lemmatize(word, "v"))
    stop_words = final_stop_words

    docs = []
    for i in dataframe[col_name]:
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', i)

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        # Convert to list from string
        text = text.split()

        # Stemming
        PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if word not in stop_words]

        text = " ".join(text)
        # print(text)
        docs.append(text)
    # print(docs)
    return docs


# Applying function
bad_reviews_data = clean_text(bad_reviews, 'review-text')
good_reviews_data= clean_text(good_reviews, 'review-text')

tab = st.sidebar.selectbox('Pick one', ['Positive Review', 'Negative Review'])

# Insert containers separated into tabs:
from bertopic import BERTopic

# Create instances of GPU-accelerated UMAP and HDBSCAN

umap_model = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, random_state= 42)

topic_model = BERTopic(language= 'en',umap_model=umap_model, n_gram_range= (1,3), verbose=True, embedding_model="all-mpnet-base-v2")

# Models
if tab == 'Positive Review':

  st.subheader('Positive Reviews')
  st.dataframe(good_reviews_data)

# Fixing small dataset bug
  if len(good_reviews) < 200: # Workaround if not enough documents https://github.com/MaartenGr/BERTopic/issues/97 , https://github.com/MaartenGr/Concept/issues/5
    good_reviews_data.extend(3*good_reviews_data)
  else:
    pass

  topic_model.fit(good_reviews_data)

else:


    # Feature Engineering
  st.subheader('Negative Reviews')
    #Accounting for small dataset

  if len(bad_reviews) < 300: # Workaround if not enough documents https://github.com/MaartenGr/BERTopic/issues/97 , https://github.com/MaartenGr/Concept/issues/5
        bad_reviews_data.extend(3*bad_reviews_data)


  st.dataframe(bad_reviews_data)
  topic_model.fit(bad_reviews_data)

topic_model.get_topic_info()

topic_labels = topic_model.generate_topic_labels(nr_words= 2)
topic_model.set_topic_labels(topic_labels)


    # pros
topic_info = topic_model.get_topic_info()

if len(good_reviews) < 300:
  topic_info['Count']=topic_info['Count']/4
  topic_info['percentage'] = topic_info['Count'].apply(lambda x: (x / topic_info['Count'].sum()) * 100)
else:
  topic_info['percentage'] = topic_info['Count'].apply(lambda x: (x / topic_info['Count'].sum()) * 100)

st.write(topic_info)
doc_num = int(st.number_input('enter the number of topic to explore', value= 0))
st.write(topic_model.get_representative_docs(doc_num))
topic_info =topic_info.to_csv(index=False).encode('utf-8')
st.download_button(
     label="Download Analysis",
     data=topic_info,
     mime='text/csv',
     file_name='analysis.csv')

'''brief_text="there is a total of {num_reviews} for the product {asin_num} of those, there are {num_en} english reviews .there are {positive_num} positive reviews and {negative_num} negative reviews.".format(
    num_reviews= len(df),
    asin_num = df['asin'].unique(),
    num_en= len(df[df['detect']=='en']),
    positive_num= len(good_reviews),
    negative_num= len(bad_reviews))'''
'''st.write(brief_text)'''