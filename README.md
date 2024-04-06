# Fake News Detection

This repository contains a web application for detecting fake news using machine learning techniques and natural language processing (NLP). Users can input news text into the web interface and receive predictions regarding its authenticity.

## Introduction

Fake news has become a significant issue in today's digital age. This project offers a solution by leveraging machine learning algorithms to classify news articles as either real or fake. The web application provides users with a simple interface to input news text and receive a prediction regarding its authenticity.

## Dataset

The dataset used in this project is sourced from [fakenews.csv](link_to_dataset), containing 4729 entries labeled as either real or fake news. It serves as the foundation for training and evaluating the machine learning models.

## Problem Statement

The primary objective of this project is to develop an accurate fake news detection model capable of distinguishing between genuine and fabricated news articles. By employing various NLP techniques and machine learning algorithms, we aim to create a reliable tool for identifying fake news.

## Features

- Exploratory Data Analysis (EDA) to understand the characteristics of the dataset.
- Preprocessing techniques including text cleaning and lemmatization.
- Implementation of multiple machine learning models for fake news detection.
- Word cloud generation to visualize frequent words in real and fake news articles.

## Exploratory Data Analysis (EDA)

The EDA section provides insights into the dataset's characteristics, including the distribution of lower and upper case letters, presence of HTML tags, URLs, hashtags, mentions, unwanted characters, and emojis.

## Preprocessing

Preprocessing steps involve text cleaning and lemmatization to prepare the text data for model training. Techniques such as removing HTML tags, URLs, hashtags, and mentions, as well as converting emojis to text, are employed to enhance the quality of the dataset.

## Models

The project implements several machine learning models for fake news detection, including:
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- K-Nearest Neighbors
- Multinomial Naive Bayes using TF-IDF
- 
Each model is trained and evaluated using appropriate metrics such as F1 score to assess its performance.

## Usage

To use the fake news detection web application:
1. Clone this repository to your local machine.
2. Install the required dependencies mentioned in the [Dependencies](#dependencies) section.
3. Run the Streamlit application using the command `streamlit run app.py`.
4. Input the news text in the provided text area and click the "Predict" button to obtain the authenticity prediction.

## Dependencies

Ensure you have the following dependencies installed:
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Wordcloud
- Textblob

You can install the dependencies using pip:
pip install -r requirements.txt


