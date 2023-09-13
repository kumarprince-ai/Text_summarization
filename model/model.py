import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.cluster.util import cosine_distance
from nltk import sent_tokenize, word_tokenize
import numpy as np
import networkx as nx
import re
import string
from google.colab import drive
drive.mount('/content/drive')

nltk.download('punkt')
nltk.download('stopwords')

import fitz #fitz is PyMuPDF


def extract_text_from_pdf(pdf_path): #extract data from pdf 
    text = ""
    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()

    return text

# Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_path = "/content/drive/MyDrive/Alldataset/Dataset for Task 5/Operations Management.pdf"
text = extract_text_from_pdf(pdf_path)
print(text)

def preprocess_text(text): #preprocess data 
    # Tokenize into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Lowercase all words
    words = [word.lower() for word in words]

    # Remove punctuation and numbers
    words = [word for word in words if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return sentences, words

def calculate_similarity_matrix(sentences, words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Calculate the similarity between sentences
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], words)

    return similarity_matrix

def sentence_similarity(sent1, sent2, words):
    vector1 = [0] * len(words)
    vector2 = [0] * len(words)

    # Build the vector representation of sentences
    for word in sent1:
        if word in words:
            vector1[words.index(word)] += 1

    for word in sent2:
        if word in words:
            vector2[words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

def generate_summary(text, num_sentences=5, max_iter=1000, alpha=0.85):
    sentences, words = preprocess_text(text)
    similarity_matrix = calculate_similarity_matrix(sentences, words)

    # Convert the similarity matrix into a graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Calculate sentence scores using PageRank with adjusted parameters
    scores = nx.pagerank(graph, max_iter= 1000, alpha=alpha)

    # Sort the sentences by their scores in descending order
    ranked_sentences = sorted(((scores[i], sent) for i, sent in enumerate(sentences)), reverse=True)

    # Select the top 'num_sentences' sentences for the summary
    summary_sentences = [sent for score, sent in ranked_sentences[:num_sentences]]

    # Detokenize the summary sentences to form the final summary
    summary = TreebankWordDetokenizer().detokenize(summary_sentences)

    return summary

if __name__ == "__main__":
    pdf_text = extract_text_from_pdf(pdf_path)

    summary = generate_summary(pdf_text, num_sentences=5)
    
    print(summary)

