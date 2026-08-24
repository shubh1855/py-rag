import json
import os
import pickle
import string

from nltk.stem import PorterStemmer

translator = str.maketrans("", "", string.punctuation)
stemmer = PorterStemmer()

with open("data/stopwords.txt", "r") as file:
    stopwords = file.read().splitlines()

stopwords = [word.lower().translate(translator) for word in stopwords]


def load_movies():
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    return data["movies"]


def tokenize_text(text):
    text = text.lower().translate(translator)

    tokens = [token for token in text.split() if token not in stopwords]

    return [stemmer.stem(token) for token in tokens]


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    def build(self):
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

    def load(self):
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)
