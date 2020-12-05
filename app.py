#%% Initial source
from flask import Flask, render_template, request, redirect, url_for, abort, Response
from flask_sqlalchemy import SQLAlchemy

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
# from bokeh.util.string import encode_utf8

import os
import pandas as pd

import numpy as np
from numpy import dot
from numpy.linalg import norm

#%% Flask start
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from models import Word2Vec, Candidate, CandidateInfo

#%% Natural language processing
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from scipy import spatial

nltk.download('stopwords')

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


#%%Bokeh example
# scatter.py

from bokeh.models import ColumnDataSource, LabelSet

num_vars = 9

theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
# rotate theta such that the first axis is at the top
theta += np.pi/2

def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def radar_patch(r, theta):
    yt = (r + 0.01) * np.sin(theta) + 0.5
    xt = (r + 0.01) * np.cos(theta) + 0.5
    return xt, yt

verts = unit_poly_verts(theta)
x = [v[0] for v in verts] 
y = [v[1] for v in verts] 

p = figure(title="Radar")
text = ['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3']
source = ColumnDataSource({'x':x+ [0.5],'y':y+ [1],'text':text})

p.line(x="x", y="y", source=source)

labels = LabelSet(x="x",y="y",text="text",source=source)

p.add_layout(labels)

# example factor:
f1 = np.array([0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00]) * 0.5
f2 = np.array([0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00]) * 0.5
f3 = np.array([0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00]) * 0.5
f4 = np.array([0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00]) * 0.5
f5 = np.array([0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]) * 0.5
#xt = np.array(x)
flist = [f1,f2,f3,f4,f5]
colors = ['blue','green','red', 'orange','purple']
for i in range(len(flist)):
    xt, yt = radar_patch(flist[i], theta)
    p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color=colors[i])

# show(p)
script, div = components(p)
# print(script)
# print(div)





#%% Flask begin
@app.route('/')
def hello():
    result = (script,div)
    return render_template('bokeh.html', result = result)
    # return "Hello World!"

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
        #Transform dataframe to dictionary to list of tuples
        result = df2.set_index('author_name').to_dict()['cos_sim']
        result = list(result.items()) 
        result = (result,script,div)

        #Return to html
        return render_template('index.html', result = result) 
    return render_template('index.html', result = {})

if __name__ == '__main__':
    app.run()