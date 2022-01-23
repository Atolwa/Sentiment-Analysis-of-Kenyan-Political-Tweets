from flask import Flask, render_template, request, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model = pickle.load(open('svm_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route("/",methods=['GET', 'POST'])
def hello_world():
    return render_template('index.html')

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    tweet = request.form.get("tweet")
    tweet_vec = vectorizer.transform([tweet])
    pred=model.predict(tweet_vec)
    return render_template('index.html',data=format(pred))
    


if __name__ == '__main__':
   app.run(debug=True)