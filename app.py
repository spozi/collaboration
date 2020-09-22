#%% Initial source
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd

import numpy as np
from numpy import dot
from numpy.linalg import norm


app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from models import Word2Vec, Candidate, CandidateInfo

#%% Natural language processing
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from scipy import spatial


lemmatizer = WordNetLemmatizer() 
tokenizer = RegexpTokenizer(r'\w+')

stop_words = list(set(stopwords.words('english')))
stop_words.extend(["et", "al", "de", "fig", "en", "use", "the"])

def textToListProcessing(text):
    lemma_text = lemmatizer.lemmatize(text) 
    tokens = tokenizer.tokenize(lemma_text.lower())
    tokens = [token for token in tokens if not token in stop_words]
    return tokens

def compute_cossim_percent(x, y):
    arrX = np.array(x.tolist())
    arrY = y
    cos_sim = 1 - spatial.distance.cosine(arrX, arrY)
    return round(cos_sim*100,2)

def authorIDtoName(x):
    res = CandidateInfo.query.filter(CandidateInfo.author_id==x).first()
    return res.author_name


#%% Flask begin
@app.route('/')
def hello():
    return "Hello World!"

@app.route('/search', methods=['GET', 'POST'])
def word_vec():
    if request.method == "POST":
        query = request.form["query"]

        #1. Vectorizing query
        tokens = textToListProcessing(query)
        vectors = []
        for token in tokens:
            res = Word2Vec.query.filter(Word2Vec.word==token).first()
            vectors.append(res.vector)

        qvector = np.mean(vectors, axis=0)  #Average the vector
        qlist = qvector.tolist()

        #2. Get a list of authors to measure the similarity (from postgresql)
        aid_res = Candidate.query.all()
        author_list = []
        for aid in aid_res:
            author_list.append((aid.id, aid.author_id, aid.vector)) 

        #Convert list to dataframe
        df = pd.DataFrame(author_list, columns=['id', 'author_id', 'vector'])
        df['cos_sim'] = df.drop(columns=['id', 'author_id']).apply(compute_cossim_percent, axis=1, args=[qlist])
        df2 = df.sort_values(by=['cos_sim'], ascending=False).drop_duplicates('author_id').sort_index()
        df2.drop(columns=['id', 'vector'], inplace=True)
        df2 = df2.sort_values(by=['cos_sim'], ascending=False)

        #Replace author_id to name
        df2['author_name'] = df2['author_id'].apply(authorIDtoName) 

        #3. Visualize
        #Transform dataframe to dictionary
        result = df2.set_index('author_name').to_dict()['cos_sim']

        #Return to html
        return render_template('index.html', result = result) 
    return render_template('index.html', result = {})

if __name__ == '__main__':
    app.run()