import requests
from bs4 import BeautifulSoup
import re
import nltk
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, render_template

nltk.download('punkt')

app = Flask(__name__)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    return tokens

def train_word2vec_model(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, sg=1, min_count=1, workers=4)
    return model

def clean_text(text):
    # Удаление нежелательных символов, сохраняя знаки препинания
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s,.!?]', '', text)
    return text

def get_page_content(url, headers):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        page_soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = page_soup.find_all('p')
        if paragraphs:
            text = ' '.join(re.sub(r'<.*?>', '', str(p)) for p in paragraphs)
            return clean_text(text.strip())
        return ""
    except Exception as e:
        print(f"Ошибка при запросе к странице: {e}")
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        results = search_and_scrape(query)
        return render_template('index.html', query=query, results=results)
    return render_template('index.html', query='', results=[])

def search_and_scrape(query):
    url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='tF2Cxc')

        documents = []
        links = []

        for result in search_results:
            title = result.find('h3').get_text()
            link = result.find('a')['href']
            
            content = get_page_content(link, headers)
            if content:
                documents.append((title, link, content))

        tokenized_docs = [preprocess_text(doc[2]) for doc in documents]

        model = train_word2vec_model(tokenized_docs)

        query_tokens = preprocess_text(query)
        query_vector = np.mean([model.wv[word] for word in query_tokens if word in model.wv], axis=0).reshape(1, -1)

        best_match = None
        highest_similarity = -1

        for i, doc in enumerate(tokenized_docs):
            doc_vector = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0).reshape(1, -1)
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = documents[i]

        return best_match if best_match else ("", "", "Не удалось найти подходящую информацию")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к Google: {e}")
        return "", "", "Ошибка при запросе к Google"

if __name__ == '__main__':
    app.run(debug=True)