#%% Hello World
# from flask import Flask, render_template, request, redirect, url_for
# import nltk
# import json
# import pandas as pd
# import gensim
# import numpy as np


# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer 
# from nltk.corpus import stopwords
# from gensim.models import KeyedVectors, phrases, Word2Vec
# import multiprocessing
# from multiprocessing import Pool
# from sklearn.metrics.pairwise import cosine_similarity

# from numpy import dot
# from numpy.linalg import norm


# authors_file = "top_10_authors.csv"
# authors_name_file = "AuthorID_Name.csv"

# #Vectorization task
# model = gensim.models.Word2Vec.load("2020.w2v")
# # print(len(model.wv.vocab))
# stop_words = list(set(stopwords.words('english')))
# stop_words.extend(["et", "al", "de", "fig", "en", "use", "the"])

# # pickle.dump(stop_words, open("stopwords.p", "wb"))

# lemmatizer = WordNetLemmatizer() 
# tokenizer = RegexpTokenizer(r'\w+')


# def textToListProcessing(text):
#     lemma_text = lemmatizer.lemmatize(text) 
#     tokens = tokenizer.tokenize(lemma_text.lower())
#     tokens = [token for token in tokens if not token in stop_words]
#     return tokens
# ##########################################################

# ############Load Existing Database########################
# #1. Using csv for now
# df = pd.read_csv(authors_file)

# # arrY = [x for x in range(0,300)]
# def compute_cossim_percent(x, y):
#     arrX = np.array(x.tolist())
#     arrY = y
#     cos_sim = dot(arrX, arrY)/(norm(arrX)*norm(arrY))
#     return round(cos_sim*100, 2)

# ############Load Author ID <-> Name File#########################
# df_auth = pd.read_csv(authors_name_file)
# auth_name_dict = df_auth.set_index('Author_ID')['Name'].to_dict()


# app = Flask(__name__)
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     author_similarity_dict = {}
#     result = {}
#     if request.method == "POST":
#         query = request.form["query"]

#         #1. Vectorizing Query
#         tokens = textToListProcessing(query)
        
#         #1a. convert tokens into vector
#         vectors = []
#         for w in tokens:
#             vectors.append(model.wv[w])
#         qvector = np.mean(vectors, axis=0)  #Average the vector
#         qlist = qvector.tolist()
        
#         #2. Get a list of author with similarity
#         df['cos_sim'] = df.drop(columns=['Unnamed: 0', 'idx', 'Author_ID']).apply(compute_cossim_percent, axis=1, args=[qlist])
#         df.sort_values(by=['cos_sim'], ascending=False, inplace=True)

#         author_similarity_dict = {}   
#         for index, row in df.iterrows():
#             if not auth_name_dict[row['Author_ID']] in author_similarity_dict:  #Check if we already have that author ID in the dictionary
#                 author_similarity_dict[auth_name_dict[row['Author_ID']]] = row['cos_sim'] #We add new author
#             elif auth_name_dict[row['Author_ID']] in author_similarity_dict:    #If the author is already there
#                 if author_similarity_dict[auth_name_dict[row['Author_ID']]] < row['cos_sim']: #Check if the author id similarity is smaller than the one that need to be update
#                     author_similarity_dict[auth_name_dict[row['Author_ID']]] = row['cos_sim'] #We update new cos_sim if previous cos_sim is smaller
#         result = author_similarity_dict
        
#         df.drop(columns=['cos_sim'], inplace=True)
#         return render_template('index.html', result = result) 
#     return render_template('index.html', result = {})

# if __name__ == '__main__':
#     app.run(debug=True)

#%% Initial source
import os
from flask import Flask


app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])

@app.route('/')
def hello():
    return "Hello World!"


@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)

if __name__ == '__main__':
    app.run()