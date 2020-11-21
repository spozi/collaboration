#%% Initial source
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd
import datetime

import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict

#Load the dataset
#1. Load stocks description and prices
stock_df =  pd.read_pickle("./assets/stocks_v1.pkl")

#2. Load stock news
stock_news_df = pd.read_excel('./assets/Malaysia_Stock_2020_Sample.xlsx', index_col=None, sheet_name='edgemarket_news')

#%% Text similarity model


#3. Load wordvec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/home/syafiq/developments/datasets/wordembedding/glove.6B.300d.txt')
tmp_file =  get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)


#%% Flask start
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
print(os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

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
            res = model[token]
            vectors.append(res)

    qvector = np.mean(vectors, axis=0)  #Average the vector
    qlist = qvector.tolist()
    return qlist

#%% Flask begin
@app.route('/')
def hello():
    return "Hello World!"

@app.route('/investhack' , methods=['GET', 'POST'])
def matching():
    if request.method == "POST":
        date_stock = request.form["dstock"]             #The required date to search matching stock with news at date
        
        shariah_stock = request.form["shariah"]         #Are we going to show only syariah compliance stock?
        selected_stock_df = {}
        if shariah_stock == 'yes':
            selected_stock_df = stock_df[stock_df['shariah'] == 'yes']
        else:
            selected_stock_df = stock_df

        quantity_stock = int(request.form["quantity"])  #How many stocks are we going to display?
        threshold = 0.85                               #This threshold can be changed to increase the specificity or sensitivity

        #1. Get the most popular stocks based on news
        #1a. Get the news 
        selected_news_df = stock_news_df[stock_news_df['date'].dt.date == datetime.datetime.strptime(date_stock,'%Y-%m-%d').date()]
        selected_news_df['lemma_title'] = selected_news_df['title'].apply(textToListProcessing)
        selected_news_df['lemma_content'] = selected_news_df['content'].apply(textToListProcessing)

        #1b. Get the stock description
        selected_stock_df['lemma_description'] = selected_stock_df['description'].apply(textToListProcessing)

        #1c. Perform matching
        #Get vector of stock description
        selected_stock_df['vector_lemma_description'] = selected_stock_df['lemma_description'].apply(tokens_to_vectors)

        #Get vector of news content
        selected_news_df['vector_lemma_title'] = selected_news_df['lemma_title'].apply(tokens_to_vectors)
        selected_news_df['vector_lemma_content'] = selected_news_df['lemma_content'].apply(tokens_to_vectors)
        # news_title = selected_news_df['title'].tolist()
        # news_content_vector = selected_news_df['vector_lemma_content'].tolist()

        #Measure the similarity between news and stock description
        popular_stock = []
        for indexX, stock in selected_stock_df.iterrows():
            for indexY, news in selected_news_df.iterrows():
                #Compute cosine similarity
                similarity = compute_cossim_percent(stock['vector_lemma_description'], news['vector_lemma_content'])
                popular_stock.append((stock['stock'], news['title'], similarity))

        #Sort based on similarity descending order
        popular_stock.sort(key=lambda tup: tup[2], reverse=True)

        #3. Remove least popular best stock for trading
        very_popular_stock = []
        for stock in popular_stock:
            if stock[2] >= threshold:
                very_popular_stock.append(stock)
        
        #4. Aggregate the popular stock, and convert it into associative rate
        agg_popular_stock = defaultdict(int)
        total_assoc = 0
        for stock in very_popular_stock:
            total_assoc += 1
            agg_popular_stock[stock[0]] += 1        #Count how many time the name of the stock occur (stock[0] is the name of the stock)

        for key, val in agg_popular_stock.items():
            agg_popular_stock[key] = round(float(val/total_assoc), 2)

        agg_popular_stock_list = [(k, v) for k, v in agg_popular_stock.items()] 

        #5. Sort Based on Popularity
        agg_popular_stock_list = sorted(agg_popular_stock_list, key=lambda x: x[1], reverse=True)
        agg_popular_stock_list = agg_popular_stock_list[:quantity_stock]

        #6. Technical Analysis
        results = []
        for stock_name, assoc_rate in agg_popular_stock_list:
            daily_price = 0
            daily_return = 0
            weekly_return = 0
            volatility = 0

            #a. Get the stock from selected_stock_df
            for index, row in selected_stock_df.iterrows():
                if row['stock'] == stock_name:
                    stock_price_df = row['history']
                    stock_price_df = stock_price_df.shift(-1, freq='D')

                    # stock_price_sr = stock_df.iloc[0]
                    # stock_price_df = stock_price_sr['history']
                    stock_price_df['daily_return'] = stock_price_df['Close'].pct_change()

                    stock_price_df['weekly_return'] = ''
                    for k in range(len(stock_price_df.index)):
                        if (k!=0) & (k%5==0):
                            stock_price_df['weekly_return'] =(stock_price_df['Close']-stock_price_df['Close'].shift(-4))/stock_price_df['Close'].shift(-4)
                    daily_price = stock_price_df['Close'][date_stock]
                    daily_return = stock_price_df['daily_return'][date_stock]
                    weekly_return = stock_price_df['weekly_return'][date_stock]

                    #b. Get the news from selected_stock_df
                    news_title = ""
                    for stk in very_popular_stock:
                        if stk[0] == stock_name:
                            news_title += "<p>" + stk[1] + "</p>"          
                    results.append((stock_name, news_title, assoc_rate, round(daily_price,4), round(weekly_return,4), round(volatility,4)))
                    break
           
        return render_template('investhack.html', result = results) 
    return render_template('investhack.html') 

if __name__ == '__main__':
    app.run()
