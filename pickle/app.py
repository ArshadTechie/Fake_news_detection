import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from wordcloud import WordCloud
from textblob import TextBlob

st.title("Fake News Detection")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\arsha\Downloads\archive (1)\fakenews.csv")
    return df

df = load_data()

# Display dataset
st.write("Dataset:", df)

# Problem Statement
if st.button("Problem Statement"):
    st.write("""
    fakenews.csv comprises 4729 unique entries, each tagged as either real or fake. This dataset serves as a cornerstone for training machine learning models dedicated to identifying fake news.

Its utility extends across various domains, including:

Real-time stream analysis: Detecting fake news within the dynamic flow of social media posts and news articles.
Natural Language Processing (NLP): Crafting algorithms to recognize linguistic patterns indicative of fabricated news.
Model interpretability: Building transparent models to shed light on the factors influencing the classification of articles as fake news.
Comparative analysis: Assessing the efficacy of diverse feature engineering methods and machine learning algorithms in fake news detection.
User empowerment: Developing applications that empower users to make informed decisions regarding the trustworthiness of news articles.
 """)

# Fake news detection functions
def eda(data):
    lower = ' '.join(data).islower()
    html = sum(1 for x in data if re.search('<.*?>', x))
    urls = sum(1 for x in data if re.search('http[s]?://.+?\S+', x))
    hasht = sum(1 for x in data if re.search('#\S+', x))
    mentions = sum(1 for x in data if re.search('@\S+', x))
    un_c = sum(1 for x in data if re.search("[]\.\*'’‘_—,:{}\-#@$%^?~`!&(0-9)]", x))
    emojiss = sum(1 for x in data if emoji.emoji_count(x))
    return lower, html, urls, hasht, mentions, un_c, emojiss

def basic_pp(x, emoj="F"):
    if emoj == "T":
        x = emoji.demojize(x)
    x = x.lower()
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http[s]?://.+?\S+', ' ', x)
    x = re.sub('#\S+', ' ', x)
    x = re.sub('@\S+', ' ', x)
    x = re.sub("[]\.\*'’‘_—,:{}\-#@$%^?~`!&(0-9)]", ' ', x)
    return x



def lemmat(x):
    sent = []
    ls = LancasterStemmer()
    for word in word_tokenize(x):
        sent.append(ls.stem(word))
    return " ".join(sent)

# Preprocessing
@st.cache_data
def preprocess_data(df):
    df['text'] = df['text'].apply(basic_pp, args=("T",))
    df['text'] = df['text'].apply(lemmat)
    return df

df = preprocess_data(df)

# Word Cloud
def generate_wordcloud(df, label):
    df_label = df[df['label'] == label]['text']
    wc = WordCloud(background_color='black', width=1600, height=800).generate(' '.join(df_label))
    st.image(wc.to_array())

# Fake News Detection Model
def fake_news_detection(x_train_p, y_train, x_test_p, y_test, model_type):
    if model_type == 'Bernoulli Naive Bayes':
        vectorizer = CountVectorizer(binary=True)
    elif model_type == 'Multinomial Naive Bayes':
        vectorizer = CountVectorizer()
    elif model_type == 'K-Nearest Neighbors':
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()

    x_train_pf = vectorizer.fit_transform(x_train_p)
    x_test_pf = vectorizer.transform(x_test_p)

    if model_type == 'Bernoulli Naive Bayes':
        model = BernoulliNB(alpha=1)
    elif model_type == 'Multinomial Naive Bayes':
        model = MultinomialNB(alpha=1)
    elif model_type=="K_Nearest_Neighbor":
        model=KNeighborsClassifier(n_neighbors=5)
    else:
        model = KNeighborsClassifier(n_neighbors=1)

    model.fit(x_train_pf, y_train)
    predicted = model.predict(x_test_pf)
    return predicted

# Prediction function
def predict_news(news_text, model_type):
    processed_text = preprocess_data(pd.DataFrame({"text": [news_text]}))
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=1, stratify=df['label'])
    predicted = fake_news_detection(x_train, y_train, processed_text['text'], None, model_type)
    return predicted[0]

# Sidebar
st.sidebar.title("Fake News Detection")

# Exploratory Data Analysis
st.sidebar.subheader("Exploratory Data Analysis")
lower, html, urls, hasht, mentions, un_c, emojiss = eda(df['text'])
st.sidebar.write("Lower and Upper Case:", lower)
st.sidebar.write("HTML Tags:", html)
st.sidebar.write("URLs:", urls)
st.sidebar.write("Hashtags:", hasht)
st.sidebar.write("Mentions:", mentions)
st.sidebar.write("Unwanted Characters:", un_c)
st.sidebar.write("Emojis:", emojiss)

# Word Cloud
st.sidebar.subheader("Word Cloud")
label = st.sidebar.radio("Select Label", ('Fake', 'Real'))
generate_wordcloud(df, 1 if label == 'Fake' else 0)

# Model Training
st.sidebar.subheader("Models")
model_type = st.sidebar.radio("Select Model", ('Bernoulli Naive Bayes', 'Multinomial Naive Bayes', 'K-Nearest Neighbors','K_Nearest_Neighbors using TFIDF', 'Multinomial Naive Bayes using TFIDF'))
if st.sidebar.button("Train Model"):
    # Splitting Data
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=1, stratify=df['label'])
    predicted = fake_news_detection(x_train, y_train, x_test, y_test, model_type)
    f1 = f1_score(y_test, predicted)
    st.sidebar.write(f"F1 Score: {f1}")

# Predict News Authenticity
st.header("News Prediction")
news_text = st.text_area("Enter the news text:")
if st.button("Predict"):
    prediction = predict_news(news_text, model_type)
    if prediction == 0:
        prediction_label = "Real"
    else:
        prediction_label = "Fake"
    st.write("Prediction:", prediction_label)
