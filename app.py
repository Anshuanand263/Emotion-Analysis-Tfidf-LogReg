import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words=set(stopwords.words('English'))
model=joblib.load('Emotion_dectector_.pkl')
vectorizer=joblib.load('TF_IDF_vectorizer.pkl')
def cleaned_text(txt):
    txt=txt.lower()#lower the txt
    txt=txt.translate(str.maketrans('','',string.punctuation))# removing punctuation
    s=''
    for c in txt:
        if not c.isdigit() and c.isascii():# removing digit and emojii
            s+=c
    words=word_tokenize(s)
    new=[]
    for tok in words:
        if tok not in stop_words:
            new.append(tok)
    txt=" ".join(new)        
    return txt
st.title('Emotion prediction App')
txt=st.text_area('Enter your feelings..')
emotion={
    0:'sadness', 1:'anger', 2:'love', 3:'surprise', 4:'fear', 5:'joy'
}
if st.button('Detect'):
    cleaned_txt=cleaned_text(txt)
    txt_vec=vectorizer.transform([cleaned_txt])
    prediction = model.predict(txt_vec)[0]
    st.success(f"The detected emotion is: {emotion[prediction]}")


