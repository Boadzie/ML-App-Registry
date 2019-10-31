import pickle
import numpy as np

#from PIL import Image
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()  # initialize it


st.markdown('# Machine Learning App Registry')
st.markdown(
    '#### These  are projects from students in the Artificial Intelligence Movement(AIM) ')


st.markdown('## App 1: VADER Sentimental Analysis')

st.write('Sentimental Analysis is a branch of Natural Language Processing \
    which involves the extraction of sentiments in text. The VADER package makes it easy to do Sentimental Analysis')

# the sentiment part


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    # return ("{} ==>  {}".format(sentence, str(score)))
    return f'The Sentiment is ==> {str(score )}'


sentence = st.text_area('Write your sentence')


if st.button('Submit'):
    result = sentiment_analyzer_scores(sentence)
    st.success(result)


st.markdown('---')
# The second app
st.markdown('## App 2: Salary Predictor For Techies')
model = pickle.load(open('model.pkl', 'rb'))  # get the model

experience = st.number_input('Years of Experience')
test_score = st.number_input('Aptitude Test score')
interview_score = st.number_input('Interview Score')

features = [experience, test_score, interview_score]


int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


if st.button('Predict'):
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f'Your Salary per anum is: Ghc {prediction[0]}')


st.markdown('---')
# The third app
# load the model
st.markdown('## App 3: Iris Flower Classifier')

# load model
iris = pickle.load(open('iris.pkl', 'rb'))

sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal Width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')

features = [sepal_length, sepal_width, petal_length, petal_width]


int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


if st.button('Report'):
    prediction = iris.predict(final_features)
    prediction = str(prediction).replace("']", '').split('-')
    st.balloons()
    st.success(f'The flower belongs to the class {prediction[1]}')
