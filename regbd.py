import pandas as pd
from bs4 import BeautifulSoup
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import class_weight

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    return text

df1 = pd.read_csv("reduced_emails.csv")
df2 = pd.read_csv("Dataset des SMS.csv")

df1 = df1.drop(columns=["Subject", "Message ID","Message","cleaned_subject","cleaned_message"], errors='ignore')
df2 = df2.rename(columns={"Category": "labels"})
df2 = df2.rename(columns={"Message": "cleaned_text"})


df1 = df1[["labels", "cleaned_text"]]
df2 = df2[["labels", "cleaned_text"]]

df = pd.concat([df1, df2], ignore_index=True)



# Define the output Excel file name
output_file = "cleaned_dataset_fin.csv"

# Save only "labels" and "cleaned_text" columns
df[["labels", "cleaned_text"]].to_csv(output_file, index=False)

print(f"Labels and cleaned text successfully saved in {output_file}")



# df['cleaned_message'] = df['Message'].apply(clean_text)

# tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
# tokenizer.fit_on_texts(df['cleaned_message'])

# df['message_tokens'] = tokenizer.texts_to_sequences(df['cleaned_message'])

# max_length = 100

# message_padded = pad_sequences(df['message_tokens'], maxlen=max_length, padding='post', truncating='post')
# df['message_padded'] = list(message_padded)

# df['label'] = df['label'].map({'spam': 1, 'ham': 0})
# labels = df['label'].values


# X_train, X_test, y_train, y_test = train_test_split(message_padded, labels, test_size=0.2, random_state=42)

# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
# class_weights = dict(enumerate(class_weights))

