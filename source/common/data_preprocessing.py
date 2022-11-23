import nltk
import numpy as np
import os
import pandas as pd
import re
import sentence_contractions

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.sparse import coo_matrix

def import_raw_data(root_dir):
    '''
    Import Chicago Hotel Review dataset
    Args:
        root_dir (str): Path for the location in google drive where data directory is located
    Returns:
        df (pd.DataFrame): A dataframe with the following columns 
            - Label (int): -1 for truthful review, 1 for deceptive review 
            - Rating (int): 1 for negative review, 5 for positive reivew
            - Ori_Review (str): Original text review without any cleaning
    '''

    file_dir = os.path.join(root_dir, 'data/raw-data')
    negative_deceptive_dir = os.path.join(file_dir, 'negative_polarity/deceptive_from_MTurk')
    negative_truthful_dir = os.path.join(file_dir, 'negative_polarity/truthful_from_Web')
    positive_deceptive_dir = os.path.join(file_dir, 'positive_polarity/deceptive_from_MTurk')
    posistive_truthful_dir = os.path.join(file_dir, 'positive_polarity/truthful_from_TripAdvisor')

    negative_deceptive_list = [file_name for file_name in os.listdir(f'{negative_deceptive_dir}') if file_name.endswith('.txt')]
    negative_truthful_list = [file_name for file_name in os.listdir(f'{negative_truthful_dir}') if file_name.endswith('.txt')]
    positive_deceptive_list = [file_name for file_name in os.listdir(f'{positive_deceptive_dir}') if file_name.endswith('.txt')]
    positive_truthful_list = [file_name for file_name in os.listdir(f'{posistive_truthful_dir}') if file_name.endswith('.txt')]

    negative_deceptive_texts = []
    negative_truthful_texts = []
    psotive_deceptive_texts = []
    psotive_truthful_texts = []
    for neg_dec_name, neg_tru_name, pos_dec_name, pos_tru_name in zip(negative_deceptive_list, negative_truthful_list, positive_deceptive_list, positive_truthful_list):
        negative_deceptive_file = open(os.path.join(negative_deceptive_dir, neg_dec_name), 'r')
        negative_truthful_file = open(os.path.join(negative_truthful_dir, neg_tru_name), 'r')
        positive_deceptive_file = open(os.path.join(positive_deceptive_dir, pos_dec_name), 'r')
        positive_truthful_file = open(os.path.join(posistive_truthful_dir, pos_tru_name), 'r')
        negative_deceptive_texts.append(negative_deceptive_file.read().strip())
        negative_truthful_texts.append(negative_truthful_file.read().strip())
        psotive_deceptive_texts.append(positive_deceptive_file.read().strip())
        psotive_truthful_texts.append(positive_truthful_file.read().strip())
        negative_deceptive_file.close()
        negative_truthful_file.close()
        positive_deceptive_file.close()
        positive_truthful_file.close()

    rating = [1] * 800 + [5] * 800
    all_text = negative_deceptive_texts + negative_truthful_texts + psotive_deceptive_texts + psotive_truthful_texts
    label = [1] * 400 + [-1] * 400 + [1] * 400 + [-1] * 400
    df = pd.DataFrame({'Label': label, 'Rating': rating, 'Ori_Review': all_text})
    return df


def get_wordnet_pos(treebank_tag):
    '''
    Get WordNet Part-of-Speech tagging
    Args:
        treebank_tag (str): string of detailed part-of-speech abbreviation
    Returns:
        Wordnet part-of-speech object grouping low-level part-of-speech to a higher level
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

    
def lemmatize(words):
    '''
    Lemmatize each token in words through WordNet's Lemmatizer.
    Ex. see, saw, seen -> see
    Args:
        words (list): list of tokens
    Returns:
        lemmatized_words (list): list of lemmatized token
    '''
    lemmatized_words = []
    lm = WordNetLemmatizer()
    for word, pos in nltk.pos_tag(words):
        lm_pos = get_wordnet_pos(pos)
        lemmatized_words.append(lm.lemmatize(word, lm_pos))
    return lemmatized_words
    
    
def remove_stopwords(words):
    '''
    Remove stopwords (ex. by, for, from, the, to) using NLTK's stopwords vocab.
    Args:
        words (list): list of tokens
    Returns:
        result (list): list of tokens with stopwords removed
    '''
    stop_words = stopwords.words('english')
    stop_words.extend(['chicago', 'hotel', 'would', 'could', 'should', 'might', 'room', 'stay'])
    result = [word for word in words if word not in stop_words]
    return result


# Remove Punctuation and turn all letter to lowercase
def preprocess_ngram(df):
    '''
    Preprocess text reviews into bag-of-words with the following cleaning process: -
        1. Remove sentence sentence contractions
        2. Remove punctuations
        3. Lemmatize words
        4. Remove stopwords
    Args:
        df (pd.DataFrame): DataFrame containing text review in column Ori_Review
    Returns
        df_pp (pd.DataFrame): DataFrame containing the same data as input, but add 3 columns
            1. Clean_Review (str): cleaned review texts
            2. PP_Review (str): cleaned review text with stopwords removed
            3. Word_List (list): list of token containing cleaned reviews with stopwords removed
    '''
    df_pp = df.copy()
    clean_review = df_pp['Ori_Review'].copy()
    clean_review = clean_review.apply(lambda x: x.lower())
    contraction_dict = sentence_contractions.get_sentence_contractions()
    for con, full in contraction_dict.items():
        clean_review = clean_review.apply(lambda x: re.sub(con, full, x))
    clean_review = clean_review.apply(lambda x: re.sub('[,/.!?]', '', x))
    words = clean_review.apply(lambda x: simple_preprocess(x, deacc=True, min_len=1))
    words = words.apply(lambda x: lemmatize(x))
    df_pp['Clean_Review'] = words.apply(lambda x: ' '.join(x))
    words = words.apply(lambda x: remove_stopwords(x))
    df_pp['PP_Review'] = words.apply(lambda x: ' '.join(x))
    df_pp['Word_List'] = words
    df_pp = df_pp.reset_index(drop=True)
    return df_pp


def preprocess_linguistic_features(df):
    df_pp = df.copy()
    clean_review = df_pp['Ori_Review'].copy()
    clean_review = clean_review.apply(lambda x: x.lower())
    contraction_dict = sentence_contractions.get_sentence_contractions()
    for con, full in contraction_dict.items():
        clean_review = clean_review.apply(lambda x: re.sub(con, full, x))
    clean_review = clean_review.apply(lambda x: re.sub('[,/.!?]', '', x))
    words = clean_review.apply(lambda x: x.split())
    words = words.apply(lambda x: lemmatize(x))
    df_pp['Word_List_all'] = words
    return df_pp


def df2matrix(df, word2ind):
    '''
    Split dataframe into X matrix and label y
    '''
    rows = []
    cols = []
    for r, document in enumerate(df['Ngram'].tolist()):
        for word in document:
            rows.append(r)
            cols.append(word2ind[word])
    vals = np.array([1] * len(rows))
    X = coo_matrix((vals, [rows, cols]), shape=(max(rows) + 1, len(word2ind))).toarray()
    X = pd.DataFrame(X, columns=list(word2ind.keys()))
    X['Rating'] = df['Rating'].copy()
    ling_fea = ['num_word', 'num_coreword', 'num_stopword', 'num_char', 'char_per_word', 'num_first_sing']
    X[ling_fea] = df[ling_fea].copy()
    y = df['Label'].copy()
    return X, y
