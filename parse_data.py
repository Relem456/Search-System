import requests
from bs4 import BeautifulSoup
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_content_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на наличие ошибок HTTP
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        list_items = soup.find_all('li')

        # Собираем весь текст из абзацев и элементов списка
        all_text = ' '.join([p.text.strip() for p in paragraphs] + [li.text.strip() for li in list_items])
        return all_text
    except requests.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

def filter_relevant_content(text, relevant_keywords=None):
    if relevant_keywords is None:
        relevant_keywords = [
            "технологии", "компьютер", "программное обеспечение", "аппаратное обеспечение",
            "ИТ", "оперативная память", "язык программирования"
        ]

    combined_text = text.lower()
    return any(keyword in combined_text for keyword in relevant_keywords)

def read_links(file_path):
    try:
        with open(file_path, 'r') as file:
            links = file.readlines()
        # Отфильтровываем пустые строки и строки с пробелами
        links = [link.strip() for link in links if link.strip()]
        return links
    except IOError as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []

def save_texts_to_file(urls, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for url in urls:
                all_text = fetch_content_from_url(url)
                if all_text and filter_relevant_content(all_text):
                    file.write(all_text + '\n\n')
    except IOError as e:
        logging.error(f"Error writing to {output_file}: {e}")

# Пример использования
if __name__ == "__main__":
    links_file = "links.txt"
    output_file = "texts_for_word2vec.txt"
    urls = read_links(links_file)
    save_texts_to_file(urls, output_file)