#%% Initial source
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd
import datetime

import numpy as np
from numpy import dot
from numpy.linalg import norm


#Load the dataset
#1. Load stocks description and prices
stock_df =  pd.read_pickle("./assets/stocks.pkl")

#2. Load stock news
stock_news_df = pd.read_excel('./assets/Malaysia_Stock_2020_Sample.xlsx', index_col=None, sheet_name='edgemarket_news')



#3. Load wordvec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/home/syafiq/developments/datasets/wordembedding/glove.6B.300d.txt')
tmp_file =  get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)


# Put temporary here
# stock_results = {}
# for index, row in stock_df.iterrows():
#     stock_name = row['stock']
#     stock_history = row['history']
#     stock_last_price = stock_history['Close'][-1]
#     # stock_results[stock_name] = stock_last_price

#%% Flask start
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
print(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# from models import Word2Vec, Candidate, CandidateInfo



#%% Natural language processing
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from scipy import spatial

nltk.download('wordnet')
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
    # arrX = np.array(x.tolist())
    arrX = x
    arrY = y
    cos_sim = 1 - spatial.distance.cosine(arrX, arrY)
    return cos_sim
    # return round(cos_sim*100,2)

def tokens_to_vectors(x):
    vectors = []
    for token in x:
        if token in model.wv:
            res = model.wv[token]
            vectors.append(res)

    qvector = np.mean(vectors, axis=0)  #Average the vector
    qlist = qvector.tolist()
    return qlist

#1. Get the most popular stocks based on news
#1a. Get the news 
selected_news_df = stock_news_df[stock_news_df['date'].dt.date == datetime.datetime.strptime('2020-01-03','%Y-%m-%d').date()]
selected_news_df['lemma_title'] = selected_news_df['title'].apply(textToListProcessing)
selected_news_df['lemma_content'] = selected_news_df['content'].apply(textToListProcessing)

#1b. Get the stock description
stock_df['lemma_description'] = stock_df['description'].apply(textToListProcessing)


#1c. Perform matching
#Get vector of stock description
stock_df['vector_lemma_description'] = stock_df['lemma_description'].apply(tokens_to_vectors)

#Get vector of news content
selected_news_df['vector_lemma_content'] = selected_news_df['lemma_content'].apply(tokens_to_vectors)
news_title = selected_news_df['title'].tolist()
news_content_vector = selected_news_df['vector_lemma_content'].tolist()


#Measure the similarity between news and stock description
popular_stock = []
for indexX, stock in stock_df.iterrows():
    for indexY, news in selected_news_df.iterrows():
        #Compute cosine similarity
        similarity = compute_cossim_percent(stock['vector_lemma_description'], news['vector_lemma_content']))
        popular_stock.append((stock['stock'], news['title'], similarity))



#%% Flask begin
@app.route('/')
def hello():
    return "Hello World!"

@app.route('/investhack' , methods=['GET', 'POST'])
def matching():
    if request.method == "POST":
        date_stock = request.form["dstock"]
        quantity_stock = request.form["quantity"]
        shariah_stock = request.form["shariah"]

        #1. Get the most popular stocks based on news
        #1a. Get the news 
        selected_news_df = stock_news_df[stock_news_df['date'].dt.date == datetime.datetime.strptime(date_stock,'%Y-%m-%d').date()]
        selected_news_df['lemma_title'] = selected_news_df['title'].apply(textToListProcessing)
        selected_news_df['lemma_content'] = selected_news_df['content'].apply(textToListProcessing)

        #1b. Get the stock description
        stock_df['lemma_description'] = stock_df['description'].apply(textToListProcessing)


        #1c. Perform matching
        #Get vector of stock description
        stock_df['vector_lemma_description'] = stock_df['lemma_description'].apply(tokens_to_vectors)

        #Get vector of news content
        selected_news_df['vector_lemma_content'] = selected_news_df['lemma_content'].apply(tokens_to_vectors)
        news_title = selected_news_df['title'].tolist()
        news_content_vector = selected_news_df['vector_lemma_content'].tolist()


        #Measure the similarity between news and stock description
        popular_stock = []
        for indexX, stock in stock_df.iterrows():
            for indexY, news in selected_news_df.iterrows():
                #Compute cosine similarity
                similarity = compute_cossim_percent(stock['vector_lemma_description'], news['vector_lemma_content']))
                popular_stock.append((stock['stock'], news['title'], similarity))

        #2. Perform financial analysis on that stock

        #3. Output the best stock for trading
        #1. Filter Shariah Stock



        #2. Match the news with given stock with frequency


        #2. Display stock and its last price
        stock_results = []
        for index, row in stock_df.iterrows():
            stock_name = row['stock']
            stock_description = row['description']
            # stock_history = row['history']
            
            stock_news = selected_news_list[index]

            #Need to check date here as date might not available
            # stock_close_price = stock_history['Close'][date_stock]
            stock_results.append((date_stock,stock_name, stock_description, stock_news))
            if(len(stock_results) > int(quantity_stock)):
                stock_results = stock_results[:-1]
                break
        #3. Select stock
        # print(stock_results)

        #3. Output the stock
        return render_template('investhack.html', result = stock_results) 
    return render_template('investhack.html') 



# @app.route('/search', methods=['GET', 'POST'])
# def word_vec():
#     if request.method == "POST":
#         query = request.form["query"]

#         #1. Vectorizing query
#         tokens = textToListProcessing(query)
#         vectors = []
#         for token in tokens:
#             res = Word2Vec.query.filter(Word2Vec.word==token).first()
#             vectors.append(res.vector)

#         qvector = np.mean(vectors, axis=0)  #Average the vector
#         qlist = qvector.tolist()

#         #2. Get a list of authors to measure the similarity (from postgresql)
#         aid_res = Candidate.query.all()
#         author_list = []
#         for aid in aid_res:
#             author_list.append((aid.id, aid.author_id, aid.vector)) 

#         #Convert list to dataframe
#         df = pd.DataFrame(author_list, columns=['id', 'author_id', 'vector'])
#         df['cos_sim'] = df.drop(columns=['id', 'author_id']).apply(compute_cossim_percent, axis=1, args=[qlist])
#         df2 = df.sort_values(by=['cos_sim'], ascending=False).drop_duplicates('author_id').sort_index()
#         df2.drop(columns=['id', 'vector'], inplace=True)
#         df2 = df2.sort_values(by=['cos_sim'], ascending=False)

#         #Replace author_id to name
#         df2['author_name'] = df2['author_id'].apply(authorIDtoName) 

#         #3. Visualize
#         #Transform dataframe to dictionary
#         result = df2.set_index('author_name').to_dict()['cos_sim']

#         #Return to html
#         return render_template('index.html', result = result) 
#     return render_template('index.html', result = {})

if __name__ == '__main__':
    app.run()
# %%
