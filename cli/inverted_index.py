import json
import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

CACHE_DIR = "cache"
BM25_K1 = 1.5
BM25_B = 0.75

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
        self.term_frequencies = {}
        self.doc_lengths = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        self.doc_lengths[doc_id] = len(tokens)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

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
        os.makedirs("CACHE_DIR", exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)

    def get_tf(self, doc_id, term):
        if doc_id not in self.term_frequencies:
            return 0

        return self.term_frequencies[doc_id].get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)

        if avg_doc_length == 0:
            length_norm = 1.0
        else:
            length_norm = (1 - b) + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)


def tokenize_term(term):
    tokens = tokenize_text(term)

    if len(tokens) != 1:
        raise Exception("Term must contain exactly one token")

    return tokens[0]
