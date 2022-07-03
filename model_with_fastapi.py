# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import pickle
import uvicorn
from fastapi import FastAPI 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from sklearn.preprocessing import StandardScaler
app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the tweet replies",
    version="0.1",
)

# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "models/TF_RF_model1.pkl"), "rb"
) as f:
    model = joblib.load(f)


# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@app.get("/predict-reply")
def predict_sentiment(reply: str):
    """
    A simple function that receive a reply content and predict the sentiment of the content.
    :param reply:
    :return: prediction, probabilities
    """
    # clean the reply
    cleaned_reply = text_cleaning(reply)
    print(cleaned_reply)
    vectorizer = pickle.load(open('vectorizers/vectorizer_with_CV.sav', 'rb'))
    print(type(vectorizer))
    text_vector = vectorizer.transform([cleaned_reply])
    print(text_vector)
    prediction = model.predict(text_vector)
    print("Prediction : ",prediction)
    output = int(prediction[0])
    probas = model.predict_proba(text_vector)
    print("Probas>>>>: ",probas[0])
    print("classes: ",model.classes_)
    i = 1 if output == 0 else 2 if output == 1 else 0
    # output_probability=probas[0][i]
    output_probability = "{:.2f}".format(float(probas[0][i]))
    print(output_probability)
    # output dictionary
    sentiments = {-1.0: "Negative", 1.0: "Positive",0.0:"Neutral"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result