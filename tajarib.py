# from zipfile import ZipFile
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os

# # Extract GloVe file
# with ZipFile('glove.6B.zip', 'r') as zip_ref:
#     zip_ref.extractall()

# # Check extracted files
# print(os.listdir())  # Ensure 'glove.6B.100d.txt' is present

# # Load GloVe embeddings
# embeddings_index = {}
# with open('glove.6B.100d.txt', encoding='utf-8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs  

# # Cosine similarity function
# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# # Create similarity matrix with missing word handling
# def create_similarity_matrix(words, embeddings_index):
#     n = len(words)
#     similarity_matrix = np.zeros((n, n))

#     for i, word1 in enumerate(words):
#         for j, word2 in enumerate(words):
#             if word1 in embeddings_index and word2 in embeddings_index:
#                 similarity_matrix[i, j] = cosine_similarity(embeddings_index[word1], embeddings_index[word2])
#             else:
#                 similarity_matrix[i, j] = 0  # If word is missing, set similarity to 0

#     return similarity_matrix

# # Use English words instead (or try FastText for French)
# words = ["start", "begin", "end", "finish", "pretty", "beautiful", "job", "work",
#          "dark", "gloomy", "open", "close"]

# # Create similarity matrix
# similarity_matrix = create_similarity_matrix(words, embeddings_index)

# # Convert to Pandas DataFrame
# df_similarity = pd.DataFrame(similarity_matrix, index=words, columns=words)

# # Plot heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(df_similarity, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Word Similarity Matrix")
# plt.show()
# Import necessary libraries
# Import necessary libraries
# Import necessary libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from bs4 import BeautifulSoup
import re
import pickle

# Example clean_text function (define it based on your preprocessing)
def clean_text(text):
    # Implement your text cleaning steps here (e.g., removing special characters, lowercasing)
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    return text

# Load your pre-trained tokenizer and model
with open("tokenizer.pkl", "rb") as tk:
    tokenizer = pickle.load(tk)  # This loads the tokenizer object

model = load_model('finemail_spam_classifier.h5')  # Replace with the actual model path

# New email data (the given text)
new_subject = ""
new_message = """
Nous avons le plaisir de vous inviter à notre événement annuel qui se tiendra le 25 mars 2025 à l'Hôtel de Ville, Paris. Cet événement réunira des experts du secteur pour discuter des dernières tendances et innovations dans le domaine de la technologie.
"""
# Clean and preprocess the new email
new_text = new_subject + ' ' + new_message
new_text_cleaned = clean_text(new_text)

# Tokenize and pad the new email
maxlen = 100  # The expected input length, make sure this matches your model's trained input shape
new_sequence = tokenizer.texts_to_sequences([new_text_cleaned])
new_padded = pad_sequences(new_sequence, maxlen=maxlen, padding='post', truncating='post')

# Make a prediction
prediction = model.predict(new_padded)

# Check if the prediction is greater than 0.5 (binary classification: spam vs not spam)
predicted_label = 'spam' if prediction[0] > 0.5 else 'not spam'

# Output the prediction
print(f"Predicted Label: {predicted_label}")
