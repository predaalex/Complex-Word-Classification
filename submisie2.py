import string
import pyphen
import numpy as np
import pandas as pd
import string

import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numpy import bincount
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from dale_chall import DALE_CHALL

dtypes = {"sentence": "string", "token": "string", "complexity": "float64"}
train = pd.read_excel('data/train.xlsx', dtype=dtypes, keep_default_na=False)
test = pd.read_excel('data/test.xlsx', dtype=dtypes, keep_default_na=False)

stop_words = set(stopwords.words('english'))

for i in range(0, len(train)):
    text = train['sentence'].iloc[i]

    tokens = word_tokenize(text)

    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    filtred_text = ""
    for word in words:
        filtred_text += word
        filtred_text += " "
    train['sentence'].iloc[i] = filtred_text

# #scoaterea semnelor de punctuatie
# for i in range(0,len(train)):
#     train['sentence'].iloc[i] = train['sentence'].iloc[i].translate(str.maketrans('', '', string.punctuation))
# impartirea datelor in date de antrenare si date de testare
# train_values, test_values, train_y, test_y = train_test_split(train, train['complex'], test_size=0.25, random_state=1000)

# sunt prea multe date de antrenare si isi ia overfill, selectez mai putin din fiecare corpus
# iar dupa reindexez df
train_values = train
# train_values = train[250:2555]
# train_values.append(train[2800:5100])
# train_values.append(train[5350:])
#
# train_values = train_values.dropna()
# train_values = train_values.reset_index()
#
# train_y = train_y.dropna()
# train_y = train_y.reset_index()
#
# test_values = test_values.dropna()
# test_values = test_values.reset_index()
#
# test_y = test_y.dropna()
# test_y = test_y.reset_index()


def is_dale_chall(word):
    if word.lower() in DALE_CHALL:
        return 0
    else:
        return 1


def is_sentence_complex(sentence):
    count_words = 0
    k = 0
    for sentence_word in sentence.split():
        count_words = count_words + 1
        if sentence_word.lower() in DALE_CHALL:
            k = k + 1

    if count_words == 0:
        return 0

    procentaj_cuvinte_dale_chall_sentence = k / count_words * 100

    if procentaj_cuvinte_dale_chall_sentence >= 30:
        return 1

    if procentaj_cuvinte_dale_chall_sentence < 30:
        return 2


def is_title(word):
    return int(word.istitle())


dic = pyphen.Pyphen(lang='en')
def nr_syllabes(word):
    syllabes = len(dic.inserted(word).split('-'))
    return syllabes


def nr_vowels(word):
    vowels = 'aeiouw'
    nr = 0
    for chr in word:
        if chr in vowels:
            nr += 1
    return nr


def suffix(word):
    noun_suffixes = ('acy', 'al', 'ance', 'dom', 'er', 'or', 'ism', 'ist', 'ity', 'ty', 'ment', 'ness', 'ship', 'sion', 'tion')
    adjective_suffixes = ('able', 'ible', 'al', 'esque', 'ful', 'ic', 'ical', 'ious', 'ous', 'ish', 'ive', 'less', 'y')
    verb_suffixes = ('ate', 'en', 'ify', 'fy', 'ize', 'ise')
    if (word.endswith(noun_suffixes)):
        return 1
    elif (word.endswith(adjective_suffixes)):
        return 2
    elif (word.endswith(verb_suffixes)):
        return 3
    else:
        return 4


def length_of_word(word):
    length = len(word)
    if (length > 8):
        return 1
    elif (length < 6):
        return 2
    else:
        return 3


nlp = spacy.load("en_core_web_lg")


def mean_vector_norm(sentence, token):
    words = nlp(sentence)

    sum_vector_norm = 0
    k = 0
    for word in words:
        k += 1
        sum_vector_norm += word.vector_norm
    medie_vector_norm_sentence = sum_vector_norm / k
    token = nlp(token)
    vector_norm_word = token.vector_norm
    difference = abs(medie_vector_norm_sentence - vector_norm_word)
    if(difference < 1):
        return 1
    else:
        return 2

###############################################################

sentences = []

for i in range(0,len(train)):
    sentence = train['sentence'].iloc[i]
    sentences.append(sentence)

# print (sentences)


categories = train['corpus'].unique()

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(sentences)

# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# training our classifier ; train_data.target will be having numbers assigned for each category in train data
clf = MultinomialNB().fit(X_train_tfidf, train['corpus'])

def word_corpus(word, corpus):
    docs_new = [(word)]
    X_new_counts = count_vect.transform(docs_new)

    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    if(predicted == corpus):
        return 1
    else:
        return 0


def get_word_structure_feature(word, sentence, corpus):
    features = []
    features.append(is_dale_chall(word))
    features.append(is_sentence_complex(sentence))
    features.append(suffix(word))
    features.append(is_title(word))
    features.append(nr_syllabes(word))
    features.append(length_of_word(word))
    features.append(nr_vowels(word))
    # features.append(word_corpus(word, corpus))
    # features.append(mean_vector_norm(sentence, word))
    return np.array(features)
from nltk.corpus import wordnet as wn


def nr_synsets(word):
    return len(wn.synsets(word))


def nr_lemmas(word):
    synsets = wn.synsets(word)
    num_lemmas = sum([len(synset.lemmas()) for synset in synsets])
    return num_lemmas


def get_wordnet_features(word, sentence):
    features = []
    features.append(nr_synsets(word))
    # features.append(nr_lemmas(word))
    return np.array(features)


def corpus_feature(corpus):
    d = {"europarl": [0], "bible": [1], "biomed": [2]}
    return d[corpus]


def featurize(row):
    word = row['token']
    sentence = row['sentence']
    corpus = row['corpus']
    all_features = []
    all_features.extend(corpus_feature(corpus))
    all_features.extend(get_wordnet_features(word, sentence))
    all_features.extend(get_word_structure_feature(word, sentence, corpus))
    return np.array(all_features)


def featurize_df(df):
    nr_of_features = len(featurize(df.iloc[0]))
    print('nr de features este: ', nr_of_features)
    nr_of_examples = len(df)
    print('nr de exemple este: ', nr_of_examples)
    features = np.zeros((nr_of_examples, nr_of_features))
    for index, row in df.iterrows():
        row_ftrs = featurize(row)
        features[index, :] = row_ftrs
    return features


# print(featurize_df(train))

x_train = featurize_df(train_values)
# print('Datele de antrenare au forma', x_train.shape)
y_train = train_values['complex'].values
# print('Etichetele datelor de antrenare arata cam asa:', y_train)

x_test = featurize_df(test)
print(x_test)
model = GaussianNB()
model.fit(x_train, y_train)
preds = model.predict(x_test)
# print(balanced_accuracy_score(test_values['complex'], preds))

df = pd.DataFrame()
df['id'] = test.index + len(train) + 1
df['complex'] = preds
df.to_csv('submission.csv', index=False)