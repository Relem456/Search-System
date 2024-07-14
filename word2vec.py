import logging
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Убедитесь, что у вас скачаны необходимые ресурсы NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Удаление нежелательных символов и приведение к нижнему регистру
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', text)  # Удаление спецсимволов
    text = text.lower()  # Приведение к нижнему регистру

    # Токенизация и удаление стоп-слов
    stop_words = set(stopwords.words('russian'))  # Используйте 'english' для английского текста
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]

    return filtered_words

def load_and_preprocess_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.read().split('\n\n')
    sentences = [preprocess_text(text) for text in texts]
    return sentences

def train_word2vec_model(texts_file, vector_size=100, window=5, min_count=5, workers=4):
    sentences = load_and_preprocess_texts(texts_file)
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model


def main():
    # Укажите путь к файлу с текстами для обучения
    texts_file = "D:\\search system\\texts_for_word2vec.txt"

    # Параметры для обучения модели Word2Vec
    vector_size = 100
    window = 5
    min_count = 5
    workers = 4

    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(
        f"Начало обучения модели Word2Vec с параметрами: vector_size={vector_size}, window={window}, min_count={min_count}, workers={workers}")

    # Обучение модели Word2Vec
    model = train_word2vec_model(texts_file, vector_size, window, min_count, workers)

    # Сохранение модели
    model.save("word2vec.model")

    logging.info("Модель Word2Vec успешно обучена и сохранена как 'word2vec.model'")

    # Пример использования модели
    try:
        similar_words = model.wv.most_similar(positive=['технология'])
        logging.info(f"Слова, похожие на 'технология': {similar_words}")
    except KeyError:
        logging.error("Слово 'технология' не найдено в модели")


if __name__ == "__main__":
    main()