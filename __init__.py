from flask import Flask, render_template, request, jsonify
import numpy as np
from gensim.models import Word2Vec

app = Flask(__name__)

# Загрузка модели Word2Vec
model = Word2Vec.load("word2vec.model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = search_using_word2vec(query)  # Функция для выполнения поиска в интернете на основе Word2Vec
    return jsonify(results)

def search_using_word2vec(query):
    query_vec = np.mean([model.wv[word] for word in query.lower().split() if word in model.wv], axis=0)
    # Вместо реального поиска в интернете возвращаем заглушку
    return [{'link': 'https://example.com', 'title': 'Example Search Result'}]

if __name__ == '__main__':
    app.run(debug=True)