import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')

# set up stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

# read all txt files from Data Set folder
folder_path = 'Data Set'
docs = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as file:
        doc = file.read()
        docs.append(doc)

# pre-process the documents
preprocessed_docs = []
for doc in docs:
    preprocessed_doc = []
    for line in doc.split('\n'):
        if ':' in line:
            line = line.split(':')[1].strip()
        words = line.split()
        words = [word for word in words if word.lower() not in stop_words]
        words = [stemmer.stem(word) for word in words]
        preprocessed_doc.extend(words)
    preprocessed_doc = ' '.join(preprocessed_doc)
    preprocessed_docs.append(preprocessed_doc)

# generate tf-idf matrix of the terms
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

query = ''
while (query != 'quit'):
    # prompt the user to enter a query
    query = input('\nEnter your query: ')

    # pre-process the query
    preprocessed_query = []
    for word in query.split():
        if word.lower() not in stop_words:
            word = stemmer.stem(word)
            preprocessed_query.append(word)
    preprocessed_query = ' '.join(preprocessed_query)

    # calculate the cosine similarity between the query and each document
    cosine_similarities = tfidf_matrix.dot(
        vectorizer.transform([preprocessed_query]).T).toarray().flatten()

    # find the index of the most similar document
    most_similar_doc_index = cosine_similarities.argsort()[::-1][0]

    # retrieve the disease information from the most similar document
    most_similar_doc = docs[most_similar_doc_index]
    disease_name = ''
    prevalence = ''
    risk_factors = ''
    symptoms = ''
    treatments = ''
    preventive_measures = ''

    for line in most_similar_doc.split('\n'):
        if line.startswith('Disease Name:'):
            disease_name = line.split(':')[1].strip()
        elif line.startswith('Prevalence:'):
            prevalence = line.split(':')[1].strip()
        elif line.startswith('Risk Factors:'):
            risk_factors = line.split(':')[1].strip()
        elif line.startswith('Symptoms:'):
            symptoms = line.split(':')[1].strip()
        elif line.startswith('Treatments:'):
            treatments = line.split(':')[1].strip()
        elif line.startswith('Preventive Measures:'):
            preventive_measures = line.split(':')[1].strip()

    # print the disease information
    print(f"\nDisease Name: {disease_name}\n")
    print(f"Prevalence: {prevalence}\n")
    print(f"Risk Factors: {risk_factors}\n")
    print(f"Symptoms: {symptoms}\n")
    print(f"Treatments: {treatments}\n")
    print(f"Preventive Measures: {preventive_measures}\n\n")
