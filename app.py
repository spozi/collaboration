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
    arrX = np.array(x)
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



#%% Universities references
univ_ref = {}
univ_ref['uum'] = [60002763,121435310,117453407,114230258,60001278,121111740,112654989,116435042]
univ_ref['upm'] = [
    60025577,60001821,60016775,114911671,60024451,106087648,115897289,60004351,106249378,60009491,
    60017880,60025530,106609727,112862476, 116759695, 107971123, 60001508, 60021145, 116278175, 122287139, 116585446
]

univ_ref['ukm'] = [60001821,60029395,119043808,110536458,60001508,118113122,112634312,60000968,119051171,109685761,120499389,
    113909037,107918306,121757963,109661456,113731863,115598983,108244260,107223852,107018029,100740780,106732676,112988341,
    115144432,119867361,120948207,118449033,60090661
]
univ_ref['um'] = [
    60029157,60002763,60013665,60025577,112612337,112115357,60002512,60021005,60020855,60017880,60001278,113670549,109906464,114359912,
    117584981,115167301,112203603
]

#Convert into dictionary of list of int to dictionary of list of str
# for k, v in univ_ref.items():
#     univ_ref[k] = list(map(str, v))

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
        university = request.form['university']

        #1. Vectorizing query
        tokens = textToListProcessing(query)
        vectors = []
        for token in tokens:
            res = Word2Vec.query.filter(Word2Vec.word==token).first()
            vectors.append(res.vector)

        qvector = np.mean(vectors, axis=0)  #Average the vector
        qlist = qvector.tolist()

        #2. Get a list of authors that belong to that selected university
        query_sql_statement = Candidate.query.filter(Candidate.afid.in_(univ_ref[university])).statement
        df_sql = pd.read_sql(query_sql_statement, Candidate.query.session.bind)

        #3. Compute the similarity
        df_sql['cos_sim'] = df_sql['description_vector'].apply(compute_cossim_percent, args=[qlist])
        df_sql.sort_values(by=['cos_sim'], ascending=False, inplace=True)
        df_sql.drop_duplicates(subset=['author_ids'], keep='first', inplace=True)
        df_sql.dropna(subset=['author_ids'], inplace=True)

        print(df_sql.head(10))
        #4. Visualize
        #Transform dataframe to dictionary to list of tuples
        result = [tuple(r) for r in df_sql[['author_names', 'cos_sim']].values]
        result = result[:10]


        #5. Return to html
        return render_template('index.html', result = result) 
    return render_template('index.html', result = {})

if __name__ == '__main__':
    app.run()