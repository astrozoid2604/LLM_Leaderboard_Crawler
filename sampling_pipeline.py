import os
import pandas as pd
import subprocess
import sys
from collections import defaultdict
import json
from tqdm import tqdm
from sklearn.decomposition import NMF
import scipy.stats
from scipy.stats import pearsonr, wasserstein_distance, ks_2samp
from scipy.special import kl_div
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from spellchecker import SpellChecker
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from gensim import corpora, models
from sklearn.manifold import TSNE
import nltk
nltk.download('punkt')
from nltk.corpus import wordnet
nltk.download('wordnet')
import torch
from transformers import BertModel, BertTokenizer
from scipy.special import kl_div
import requests
import spacy
import time
#### Please pip install --->>> `pip install textstat==0.6.2` and python -m spacy download en_core_web_sm

from textstat.textstat import textstatistics,legacy_round
global nlp
nlp = spacy.load('en_core_web_sm')
nlp.disable_pipe("parser")
nlp.add_pipe("sentencizer")



cwd_path = os.getcwd()

max_input_token_len = 2000

                      #------------------- MMLU SEARCH KEYWORD -------------------- : ---- BEST SAMPLING METHOD PER MMLU SUBJECT ---#
mmlu_subject_dict = {
                      'harness|hendrycksTest-high_school_government_and_politics|5':'random',
                                         'harness|hendrycksTest-abstract_algebra|5':'clustering_Spectral_MTEB',
                                                  'harness|hendrycksTest-anatomy|5':'clustering_Spectral_MTEB',
                                                'harness|hendrycksTest-astronomy|5':'random',
                                          'harness|hendrycksTest-business_ethics|5':'quality_compound_probability_distribution',
                                       'harness|hendrycksTest-clinical_knowledge|5':'clustering_Spectral_MTEB',
                                          'harness|hendrycksTest-college_biology|5':'quality_spelling_error',
                                        'harness|hendrycksTest-college_chemistry|5':'quality_compound_probability_distribution',
                                 'harness|hendrycksTest-college_computer_science|5':'quality_compound_probability_distribution',
                                      'harness|hendrycksTest-college_mathematics|5':'clustering_Spectral_MTEB',
                                         'harness|hendrycksTest-college_medicine|5':'clustering_Spectral_BERT',
                                          'harness|hendrycksTest-college_physics|5':'clustering_Spectral_BERT',
                                        'harness|hendrycksTest-computer_security|5':'clustering_NMF_TFIDF',
                                       'harness|hendrycksTest-conceptual_physics|5':'clustering_Spectral_BERT',
                                             'harness|hendrycksTest-econometrics|5':'clustering_NMF_TFIDF',
                                   'harness|hendrycksTest-electrical_engineering|5':'quality_spelling_error',
                                   'harness|hendrycksTest-elementary_mathematics|5':'quality_lexical_diversity',
                                             'harness|hendrycksTest-formal_logic|5':'clustering_Spectral_BERT',
                                             'harness|hendrycksTest-global_facts|5':'quality_compound_probability_distribution',
                                      'harness|hendrycksTest-high_school_biology|5':'clustering_Spectral_MTEB',
                                    'harness|hendrycksTest-high_school_chemistry|5':'quality_compound_probability_distribution',
                             'harness|hendrycksTest-high_school_computer_science|5':'quality_spelling_error',
                             'harness|hendrycksTest-high_school_european_history|5':'clustering_Spectral_BERT',
                                    'harness|hendrycksTest-high_school_geography|5':'clustering_NMF_TFIDF',
                               'harness|hendrycksTest-high_school_macroeconomics|5':'clustering_NMF_TFIDF',
                                  'harness|hendrycksTest-high_school_mathematics|5':'clustering_NMF_TFIDF',
                               'harness|hendrycksTest-high_school_microeconomics|5':'quality_spelling_error',
                                      'harness|hendrycksTest-high_school_physics|5':'quality_spelling_error',
                                   'harness|hendrycksTest-high_school_psychology|5':'random',
                                   'harness|hendrycksTest-high_school_statistics|5':'clustering_NMF_TFIDF',
                                   'harness|hendrycksTest-high_school_us_history|5':'quality_spelling_error',
                                'harness|hendrycksTest-high_school_world_history|5':'clustering_KMeans_TFIDF',
                                              'harness|hendrycksTest-human_aging|5':'random',
                                          'harness|hendrycksTest-human_sexuality|5':'clustering_Spectral_BERT',
                                        'harness|hendrycksTest-international_law|5':'quality_spelling_error',
                                            'harness|hendrycksTest-jurisprudence|5':'clustering_NMF_TFIDF',
                                        'harness|hendrycksTest-logical_fallacies|5':'random',
                                         'harness|hendrycksTest-machine_learning|5':'quality_spelling_error',
                                               'harness|hendrycksTest-management|5':'clustering_Spectral_BERT',
                                                'harness|hendrycksTest-marketing|5':'clustering_KMeans_TFIDF',
                                         'harness|hendrycksTest-medical_genetics|5':'quality_lexical_diversity',
                                            'harness|hendrycksTest-miscellaneous|5':'clustering_NMF_TFIDF',
                                           'harness|hendrycksTest-moral_disputes|5':'random',
                                          'harness|hendrycksTest-moral_scenarios|5':'clustering_NMF_TFIDF',
                                                'harness|hendrycksTest-nutrition|5':'clustering_Spectral_BERT',
                                               'harness|hendrycksTest-philosophy|5':'quality_spelling_error',
                                               'harness|hendrycksTest-prehistory|5':'quality_lexical_diversity',
                                  'harness|hendrycksTest-professional_accounting|5':'random',
                                         'harness|hendrycksTest-professional_law|5':'clustering_NMF_TFIDF',
                                    'harness|hendrycksTest-professional_medicine|5':'clustering_Spectral_MTEB',
                                  'harness|hendrycksTest-professional_psychology|5':'quality_compound_probability_distribution',
                                         'harness|hendrycksTest-public_relations|5':'clustering_KMeans_TFIDF',
                                         'harness|hendrycksTest-security_studies|5':'clustering_KMeans_TFIDF',
                                                'harness|hendrycksTest-sociology|5':'quality_spelling_error',
                                        'harness|hendrycksTest-us_foreign_policy|5':'clustering_NMF_TFIDF',
                                                 'harness|hendrycksTest-virology|5':'clustering_Spectral_MTEB',
                                          'harness|hendrycksTest-world_religions|5':'quality_compound_probability_distribution',
                    }

filter_dict = {                                               
                      'arc': ['harness|arc:challenge|25', 'acc_norm'],
               'truthfulqa': [ 'harness|truthfulqa:mc|0',      'mc2'],
               'winogrande': [    'harness|winogrande|5',      'acc'],
                    'gsm8k': [         'harness|gsm8k|5',      'acc'],
                'hellaswag': [    'harness|hellaswag|10', 'acc_norm'],
                     'mmlu': [  mmlu_subject_dict.keys(), 'acc_norm'],
              }
for mmlu_subject in mmlu_subject_dict.keys():
    filter_dict.update({mmlu_subject.split("|")[1]:[mmlu_subject, 'acc_norm']})

similarity_measures_list = ['Pearson Coefficient',
                            'KL Divergence',
                            'Wasserstein Distance',
                            'Kolmogorov-Smirnov Test',
                           ]

vowels, consonants = 'aeiou', 'bcdfghjklmnpqrstvwxyz'



def calculate_avg_score(json_filepath, benchmark_of_interest):
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # When there is at least 1 test not being executed (hence missing score), avg_score=0 will be returned
    avg_score = 0.0
    try:
        search_keyword   = filter_dict[benchmark_of_interest][0]
        metric_name      = filter_dict[benchmark_of_interest][1]
        if 'mmlu'==benchmark_of_interest:
           mmlu_acc_norm_avg  = np.mean([data['results'][key]['acc_norm'] for key in search_keyword]) 
        else:
            final_score_list = [float(data['results'][search_keyword][metric_name])]
        avg_score = np.mean(final_score_list)
    except: pass

    # Need to skip NaN test score from result.json file
    return 0.0 if np.isnan(avg_score) else avg_score


###################################################################################
###################### Helper Functions for Quality Sampling ######################
###################################################################################

def spelling_check(string):
    wordlist=string.split()
    spell = SpellChecker(case_sensitive=False)
    spell.distance = 1
    amount_miss = len(list(spell.unknown(wordlist)))
    return amount_miss #Type: int


def avg_length_of_words(string):
    words = string.split()
    return sum(len(word) for word in words)/len(words)


def count_number_of_period(string):
    number_of_period = string.count('.')
    return number_of_period


def vowels_to_consonants_ratio(string):
    wordlist=string.split()
    vowel_count = 0
    consonant_count = 0
    for word in wordlist:
        for letter in word:
            if letter in vowels:
                vowel_count += 1
            elif letter in consonants:
                consonant_count += 1
    return vowel_count/consonant_count


def wordform(string):
    wordlist=string.split()
    uppercase_words = 0
    lowercase_words = 0
    for word in wordlist:
        if word.isupper():
            uppercase_words += 1
        elif word.islower():
            lowercase_words += 1
    return uppercase_words/lowercase_words
    

def compound_abbreviation_score(wordform,number_of_period,vowels_to_consonants_ratio):
    compound = (wordform+number_of_period+vowels_to_consonants_ratio)/3
    return compound


def compound_probability_distribution(df_full):
    compound_score = df_full['compound_abbr_score'].to_list()
    compound_score.sort()
    compound_probability_distribution = []
    for i in range(len(compound_score)):
        compound_probability_distribution.append(i/len(compound_score))
    return compound_probability_distribution


def lexical_diversity(string):
    wordlist = string.split()
    return len(set(wordlist))/len(wordlist)


def count_grammatically_incorrect_to_correct(string):
    wordlist = string.split()
    matches = language_tool_python.LanguageTool('en-US').check(string)
    return len(matches)/len(wordlist)


def count_repeating_words(string):
    words = string.split()
    counts = {}
    for word in words:
        if word not in counts:
            counts[word] = 0
        counts[word] += 1
    return sum(counts.values()) / len(words)
    

###################################################################################
############################ End of Quality Sampling ##############################
###################################################################################

###################################################################################
################## Helper Functions for Difficulty based Sampling #################
###################################################################################

# Splits the text into sentences, using 
# Spacy's sentence segmentation which can 
# be found at https://spacy.io/usage/spacy-101
def break_sentences(text):
    doc = nlp(text)
    return list(doc.sents)
 
# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words
 
# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)
 
# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length
 
# Textstat is a python package, to calculate statistics from 
# text to determine readability, 
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
    return textstatistics().syllable_count(word)
 
# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)
 
# Return total Difficult Words in a text
def difficult_words(text) -> int:
     
    
    doc = nlp(text)
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
 
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as 
    # a list of common words
    diff_words_set = set()
     
    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)
    return len(diff_words_set)
 
# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words 
# present in the text
def poly_syllable_count(text):
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]
     
 
    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count
 
 
def flesch_reading_ease(text):
    """
        Implements Flesch Formula:
        Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        Here,
          ASL = average sentence length (number of words 
                divided by number of sentences)
          ASW = average word length in syllables (number of syllables 
                divided by number of words)
    """
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -\
          float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)
 
 
def gunning_fog(text):
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade
    
def smog_index(text):
    """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here, 
           polysyllable count = number of words of more
          than two syllables in a sample of 30 sentences.
    """
    
    if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        SMOG = (1.043 * (30*(poly_syllab / sentence_count(text)))**0.5) \
                + 3.1291
        return legacy_round(SMOG, 1)
    else:
        return 0
 
 
def dale_chall_readability_score(text):
    """
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
    """
    words = word_count(text)
    
    # Number of words not termed as difficult words
    diff_word_count = difficult_words(text)
    count = words - diff_word_count

    per = 0
    if words > 0:
 
        # Percentage of words not on difficult word list
 
        per = float(count) / float(words) * 100
     
    # diff_words stores percentage of difficult words
    diff_words = 100 - per
 
    raw_score = (0.1579 * diff_words) + \
                (0.0496 * avg_sentence_length(text))
     
    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score
 
    if diff_words > 5:       
 
        raw_score += 3.6365
         
    return legacy_round(raw_score, 2)

def difficult_words_count(text):
    words = word_count(text)
    
    # Number of words not termed as difficult words
    diff_word_count = difficult_words(text)

    return diff_word_count

def percentage(data,min,max):
    return (data - min) / (max - min) 


######################################################################################
############################ End of Difficulty sampling ##############################
######################################################################################
    

def get_col_of_interest(df_full, metric):
    cols_of_interest = [metric, 'metrics.'+metric, "metrics"]
    target_col = ''
    for col in cols_of_interest:
        if col in df_full.columns: 
            target_col = col
            break
    return target_col


def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings #Type: <class 'numpy.ndarray'>


def get_mteb_embedding(input_text, endpoint_url="http://15.116.85.48:7171/embed"):

    try:
        input_data = {"inputs": input_text[:max_input_token_len]}
        input_json = json.dumps(input_data)
        headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint_url, data=input_json, headers=headers)

        if response.status_code == 200:
            result = json.loads(response.text)
            return result[0]  # Assuming the endpoint returns a single embedding
                              # Type: <class 'list'>
        else:
            print(f"Request failed with status code: {response.status_code}. Restarting with 50% lower max_input_token_len={0.5*max_input_token_len}")
            time.sleep(3)
            input_data = {"inputs": input_text[:int(max_input_token_len*0.5)]}
            response = requests.post(endpoint_url, data=json.dumps(input_data), headers={"Content-Type": "application/json"})
            result = json.loads(response.text)
            return result[0]  # Assuming the endpoint returns a single embedding
                              # Type: <class 'list'>

    except Exception as e:
        print(f"Error sending request: {str(e)}")
        ans = None
        it = 0
        while(True):
            it+=1
            try:
                time.sleep(5)
                input_data = {"inputs": input_text[:int(max_input_token_len*0.01)]}
                response = requests.post(endpoint_url, data=json.dumps(input_data), headers={"Content-Type": "application/json"})
                result = json.loads(response.text)
                ans = result[0]  # Assuming the endpoint returns a single embedding
                                 # Type: <class 'list'>
                print(f"\tRetry session {it} successfully re-established connection")
                break            
            except:
                print(f"\tRetry session {it} failed to establish connection")
        return ans


def convert_bool_to_int(x):
    if x=="True": return 1.0
    else:         return 0.0


def calc_difficult_words_percentage(df_full):
    col = 'question' if 'question' in df_full.columns else 'example'
    df_full["difficult_word_count"] = df_full[col].apply(difficult_words_count)
    min_diff = min(df_full.difficult_word_count)
    max_diff = max(df_full.difficult_word_count)
    df_full["difficult_words_percentage"] = df_full.difficult_word_count.apply(lambda data: percentage(data,min_diff,max_diff))
    return df_full


def calc_difficult_DaleChall_Readability(df_full):
    col = 'question' if 'question' in df_full.columns else 'example'
    df_full["difficult_DaleChall_Readability"] = df_full[col].apply(dale_chall_readability_score)
    return df_full


def calc_difficult_Flesch_Readability(df_full):
    col = 'question' if 'question' in df_full.columns else 'example'
    df_full['difficult_Flesch_Readability'] = df_full[col].apply(flesch_reading_ease)
    return df_full


def calc_difficult_Gunning_Fog(df_full):
    col = 'question' if 'question' in df_full.columns else 'example'
    df_full['difficult_Gunning_Fog'] = df_full[col].apply(gunning_fog)
    return df_full


def calc_quality_spelling_error(df_full, spelling_error_list, saved_expensive_list):
    if len(spelling_error_list)==0:
        spelling_error_list = df_full['full_prompt'].apply(spelling_check).tolist()
        saved_expensive_list[2] = spelling_error_list
    df_full['spelling_error']                           = pd.Series(spelling_error_list)
    return df_full, spelling_error_list, saved_expensive_list


def calc_quality_avg_length_of_words(df_full):
    df_full['avg_length_of_words']                      = df_full['full_prompt'].apply(avg_length_of_words)
    return df_full


def calc_quality_compound_probability_distribution(df_full):
    df_full['vowels_to_consonants_ratio']               = df_full['full_prompt'].apply(vowels_to_consonants_ratio)
    df_full['wordform']                                 = df_full['full_prompt'].apply(wordform)
    df_full['number_of_period']                         = df_full['full_prompt'].apply(count_number_of_period)
    df_full['compound_abbr_score']                      = df_full.apply(lambda x: compound_abbreviation_score(x["wordform"],x["number_of_period"],x["vowels_to_consonants_ratio"]),axis=1)
    df_full['compound_probability_distribution']        = compound_probability_distribution(df_full)
    return df_full


def calc_quality_lexical_diversity(df_full):
    df_full['lexical_diversity']                        = df_full['full_prompt'].apply(lexical_diversity)
    return df_full


def calc_quality_count_repeating_words(df_full):
    df_full['count_grammatically_incorrect_to_correct'] = df_full['full_prompt'].apply(count_grammatically_incorrect_to_correct)
    df_full['count_repeating_words']                    = df_full['full_prompt'].apply(count_repeating_words)    
    return df_full


def calc_quality_number_of_period(df_full):
    df_full['number_of_period']                         = df_full['full_prompt'].apply(count_number_of_period)
    return df_full


def calc_clustering_NMF_TFIDF(df_full, sample_random_state, is_adaptive=False):
    dtm_full_prompt = TfidfVectorizer(max_df=0.75, min_df=2, stop_words='english').fit_transform(df_full['full_prompt'])
    topic_results = NMF(n_components=20,random_state=sample_random_state).fit(dtm_full_prompt).transform(dtm_full_prompt)
    cluster_keyword = 'Cluster' if is_adaptive==False else 'Cluster_NMF_TFIDF'
    df_full[cluster_keyword] = topic_results.argmax(axis=1)
    return df_full


def calc_clustering_LDA_TFIDF(df_full, sample_random_state, is_adaptive=False):
    col = 'question' if 'question' in df_full.columns else 'example'
    tokenized_data = [text.split() for text in df_full[col].tolist()]
    dictionary = corpora.Dictionary(tokenized_data)
    #dictionary.filter_extremes(no_below=3, no_above=0.7) #This causes the dictionary to be empty
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    topics = models.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary, passes=20, iterations=100, alpha='auto', eta='auto')[corpus_tfidf]
    cluster_keyword = 'Cluster' if is_adaptive==False else 'Cluster_LDA_TFIDF'
    df_full[cluster_keyword] = pd.Series([max(topic, key=lambda x: x[1])[0] for topic in topics])
    return df_full


def calc_clustering_KMeans_TFIDF(df_full, sample_random_state, is_adaptive=False):
    col = 'question' if 'question' in df_full.columns else 'example'
    num_clusters = 8 #Based on the result of ELBOW finding by Parwez
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(df_full[col])
    tsne_result = TSNE(n_components=2, random_state=sample_random_state,method='barnes_hut', init='random').fit_transform(tfidf_matrix)
    cluster_keyword = 'Cluster' if is_adaptive==False else 'Cluster_KMeans_TFIDF'
    df_full[cluster_keyword] = KMeans(n_clusters=num_clusters, random_state=sample_random_state).fit_predict(tsne_result)
    return df_full


def calc_clustering_Spectral_BERT(df_full, bert_embeddings, saved_expensive_list, sample_random_state, is_adaptive=False):
    col = 'question' if 'question' in df_full.columns else 'example'
    num_clusters = 17  #Based on the result of ELBOW finding by Parwez
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    if len(bert_embeddings)==0:
        bert_embeddings = np.array([get_bert_embedding(text, tokenizer, bert_model) for text in tqdm(df_full[col].tolist(), desc="Bert Embedding...", ascii=False, ncols=75)])
        saved_expensive_list[0] = bert_embeddings
    cluster_keyword = 'Cluster' if is_adaptive==False else 'Cluster_Spectral_BERT'
    df_full[cluster_keyword] = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=sample_random_state).fit_predict(bert_embeddings)
    return df_full, bert_embeddings, saved_expensive_list


def calc_clustering_Spectral_MTEB(df_full, MTEB_embeddings, saved_expensive_list, sample_random_state, is_adaptive=False):
    col = 'question' if 'question' in df_full.columns else 'example'
    num_clusters_new = 26  #Based on the result of ELBOW finding by Parwez
    if len(MTEB_embeddings)==0:
        MTEB_embeddings = np.array([get_mteb_embedding(text) for text in tqdm(df_full[col].tolist(), desc="MTEB Embedding...", ascii=False, ncols=75)])
        saved_expensive_list[1] = MTEB_embeddings
    cluster_keyword = 'Cluster' if is_adaptive==False else 'Cluster_Spectral_MTEB'
    df_full[cluster_keyword] = SpectralClustering(n_clusters=num_clusters_new, affinity='nearest_neighbors', random_state=sample_random_state).fit_predict(MTEB_embeddings)
    return df_full, MTEB_embeddings, saved_expensive_list


def calc_clustering_DBSCAN_TFIDF(df_full, sample_random_state, is_adaptive=False):
    col = 'question' if 'question' in df_full.columns else 'example'
    tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(df_full[col])
    cosine_sim = cosine_similarity(tfidf_matrix)
    cluster_keyword = 'Cluster' if is_adaptive==False else 'Cluster_DBSCAN_TFIDF'
    df_full[cluster_keyword] = DBSCAN(eps=0.7, min_samples=2, metric='cosine').fit_predict(cosine_sim)
    return df_full


def calc_random(df_full):
    return df_full


sampling_method_dict     = {
                            #          'difficult_DaleChall_Readability':calc_difficult_DaleChall_Readability,           # Readability metric, harder has higher scores, easier has lower scores 
                            #             'difficult_Flesch_Readability':calc_difficult_Flesch_Readability,              # Readability metric, harder has lower score, easier has higher score
                            #                    'difficult_Gunning_Fog':calc_difficult_Gunning_Fog,                     # Readability metric, easier has lower scores, harder has higher score
                            #               'difficult_words_percentage':calc_difficult_words_percentage,                # Per Gayathri, percentage of difficult words/ total words in a sentence
                            #              'quality_avg_length_of_words':calc_quality_avg_length_of_words,
                                                                                                                      
                                                               'random':calc_random,                                    # Short  run ~27 models per sec
                            'quality_compound_probability_distribution':calc_quality_compound_probability_distribution, # Short  run ~13 models per sec
                                            'quality_lexical_diversity':calc_quality_lexical_diversity,                 # Short  run ~21 models per sec
                                               'quality_spelling_error':calc_quality_spelling_error,                    # Long   run ~1.00 min  (fixed) + 10 models per sec
                                                 'clustering_NMF_TFIDF':calc_clustering_NMF_TFIDF,                      # Short  run ~ 9 models per sec 
                                                 'clustering_LDA_TFIDF':calc_clustering_LDA_TFIDF,                      # Medium run ~2.00 secs per model
                                              'clustering_KMeans_TFIDF':calc_clustering_KMeans_TFIDF,                   # Medium run ~2.00 secs per model
                                              'clustering_Spectral_MTEB':calc_clustering_Spectral_MTEB,                   # Long   run ~3.50 mins (fixed) +  4 models per sec
                                             'clustering_Spectral_BERT':calc_clustering_Spectral_BERT,                  # Long   run ~0.75 mins (fixed) +  2 models per sec
                                                                                                                       
                            #                 'quality_number_of_period':calc_quality_number_of_period,
                            #            'quality_count_repeating_words':calc_quality_count_repeating_words,             # Per Gayathri, we can ignore this quality indicator for now, as we will be proceeding with embedding based where we might use similarity index, which might be indicative of duplication 
                            #                  'clustering_DBSCAN_TFIDF':calc_clustering_DBSCAN_TFIDF,                   # Per Parwez,   we can ignore this clustering method for now as he needs to further fine-tune it to get comparable result as LDA and KMeans
                            }


def calc_adaptive(df_full, bert_embeddings, MTEB_embeddings, spelling_error_list, saved_expensive_list, sample_random_state):
    enabled_sampling_method_list = list(sampling_method_dict.keys())
    enabled_sampling_method_list.remove('adaptive') # Removing adaptive sampling_method to avoid endless recursive on calc_adaptive() function
    for sampling_method in enabled_sampling_method_list:
        if sampling_method=='random': 
            df_full = sampling_method_dict[sampling_method](df_full)
        
        elif 'clustering_' in sampling_method:
            if sampling_method=='clustering_Spectral_MTEB':
                df_full, MTEB_embeddings, saved_expensive_list = sampling_method_dict[sampling_method](df_full, MTEB_embeddings, saved_expensive_list, sample_random_state, is_adaptive=True)
            elif sampling_method=='clustering_Spectral_BERT':
                df_full, bert_embeddings, saved_expensive_list = sampling_method_dict[sampling_method](df_full, bert_embeddings, saved_expensive_list, sample_random_state, is_adaptive=True)
            else:
                df_full = sampling_method_dict[sampling_method](df_full, sample_random_state, is_adaptive=True)
        
        elif 'quality_' in sampling_method:
            if sampling_method=='quality_spelling_error':
                df_full, spelling_error_list, saved_expensive_list = sampling_method_dict[sampling_method](df_full, spelling_error_list, saved_expensive_list)
            else:
                df_full = sampling_method_dict[sampling_method](df_full)
            df_full['quality_asc']   = df_full['sampling_method'].apply(lambda x: False if x[len('quality_'):] in ["compound_probability_distribution","avg_length_of_words","lexical_diversity"] else True)
        
        elif 'difficult_' in sampling_method:
            col = 'question' if 'question' in df_full.columns else 'example'
            df_full = sampling_method_dict[sampling_method](df_full, col)
            df_full['difficult_asc'] = df_full['sampling_method'].apply(lambda x: True if x in ['difficult_Flesch_Readability'] else False)

    return df_full, bert_embeddings, MTEB_embeddings, spelling_error_list, saved_expensive_list 


def get_adaptive_df_sample(df_full, sample_pct, sample_random_state):
    mmlu_subject_list = list(mmlu_subject_dict.keys())
    df_sample_chunk_list = []

    for mmlu_subject in mmlu_subject_list:
        full_chunk = pd.DataFrame()
        full_chunk = df_full[df_full['mmlu_subject']==mmlu_subject]
        sampling_method = full_chunk['sampling_method'].tolist()[0]

        if sampling_method=='random': 
            sample_chunk = full_chunk.sample(frac=sample_pct, random_state=sample_random_state)
        elif 'clustering_' in sampling_method:
            cluster_keyword = 'Cluster_'+sampling_method[len('clustering_'):]
            sample_chunk = full_chunk.groupby(cluster_keyword, group_keys=False).apply(lambda x: x.sample(int(np.rint((sample_pct*len(df_full))*len(x)/len(df_full))), random_state=sample_random_state))#.sample(frac=1)
        elif 'quality_' in sampling_method:
            quality_asc   = full_chunk['quality_asc'].tolist()[0]
            sample_chunk  = full_chunk.sort_values([sampling_method[len('quality_'):]], ascending=quality_asc, axis=0).head(int(full_chunk.shape[0]*sample_pct))
        elif 'difficult_' in sampling_method:
            difficult_asc = full_chunk['difficult_asc'].tolist()[0] 
            sample_chunk  = full_chunk.sort_values([sampling_method], ascending=difficult_asc, axis=0).head(int(full_chunk.shape[0]*sample_pct))
        df_sample_chunk_list += [sample_chunk]

    df_sample = pd.concat(df_sample_chunk_list, axis=0, ignore_index=True)
    return df_sample


def get_top_models(num_top_models=10, benchmark_of_interest='all', LLM_list=[], model_sampling_interval=1):
    if num_top_models<=0:
        print("\nPlease input a strictly positive number of best model to detect")
        return []
    if len(LLM_list)>0: num_top_models = int(len(LLM_list))

    if benchmark_of_interest=='mmlu' and 'adaptive' not in sampling_method_dict.keys(): sampling_method_dict.update({'adaptive':calc_adaptive})
    if benchmark_of_interest!='mmlu' and 'adaptive'     in sampling_method_dict.keys(): sampling_method_dict.pop('adaptive')

    # Cloning evaluation result for all models captured in HuggingFace's LLM leaderboard
    os.chdir(cwd_path)
    try:    subprocess.run(['git', 'clone', 'https://huggingface.co/datasets/open-llm-leaderboard/results'])
    except: pass

    # Retreving all augs, models, and result.json files
    os.chdir(f'{cwd_path}/results/')
    filenames = subprocess.run(["git", "ls-tree", "--full-name", "--name-only", "-r", "HEAD"], capture_output = True, text = True).stdout.split('\n')
    filenames = [filename for filename in filenames if 'json' in filename]
    os.chdir(cwd_path)

    # Outer dict key is aug, Inner dict key is model, Value is list of tuples
    #   Each tuple corresponds to different evaluation timestamps for same aug and same model
    #   Each tuple comprises of directory for timestamp, and the average score from 6 LLM benchmarks (ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K)
    aug_model_result_dict  = defaultdict(dict)

    for i, filename in tqdm(enumerate(filenames), desc="Traversing HuggingFace LLM Leaderboard results...", ascii=False, ncols=75):
        # If we explicitly specify the list of models, we just need to traverse on said models of interest. 
        # No need to traverse every single model in HuggingFace results/
        if len(LLM_list)>0:
            found_flag = False
            for LLM in LLM_list:
                if LLM in filename:
                    found_flag = True
                    break
            if not found_flag: continue
        
        # If there is no result.json file available for a particular aug or model, continue to next iteration 
        try: [aug, model, result] = filename.split('/')
        except: continue
            
        timedir = result[8: -5]

        # When any of aug, model, time directory, or result.json substring is missing, the average score is outdated and unreliable.
        #   In such case, let's just skip it
        if len(aug)==0 or len(model)==0 or len(timedir)==0 or len(result)==0: continue


        model_result_dict = defaultdict(list)

        avg_score = calculate_avg_score(f'{cwd_path}/results/{aug}/{model}/results_{timedir}.json', benchmark_of_interest)
        model_result_dict[model] += [(timedir, avg_score)]

        if aug not in aug_model_result_dict.keys():
            aug_model_result_dict[aug] = model_result_dict
        else: 
            aug_model_result_dict[aug][model] += [(timedir, avg_score)]

    # Sorting the aug:model:result dictionary in descending order based on average score
    for aug_key in aug_model_result_dict.keys():
        for model_key in aug_model_result_dict[aug_key].keys():
            aug_model_result_dict[aug_key][model_key] = sorted(aug_model_result_dict[aug_key][model_key], key=lambda val: val[1], reverse=True) # Sort by score 

    # Getting list of top models based on average score sorted in descending order
    #   Example: [(77.29, 'moreh/MoMo-70B-lora-1.8.6-DPO/2024-01-16T21-53-27.045677'),]
    top_models = [(float(f'{aug_model_result_dict[aug_key][model_key][0][1]*100:.2f}'), aug_key+'/'+model_key+'/'+aug_model_result_dict[aug_key][model_key][0][0]) for aug_key in aug_model_result_dict.keys() for model_key in aug_model_result_dict[aug_key].keys()] 
    top_models = sorted(top_models, key=lambda val: val[0], reverse=True)
    
    del aug_model_result_dict
    return top_models[:num_top_models*model_sampling_interval:model_sampling_interval], benchmark_of_interest


def tabulate_score(top_models, benchmark_of_interest='all', sampling_method='random', sample_pct=.25, saved_df_full=[pd.DataFrame()], problematic_model_pqt=[], saved_expensive_list=[[], [], []]):
    assert len(top_models)>0, "No models are captured"
    
    metrics_dict = {filter_dict[benchmark_of_interest][1]: []} # List content: [aug/model, full-data score, sample-data score]
    sample_random_state = 1
    
    resume_df_full = []
    
    os.chdir(cwd_path)

    bert_embeddings, MTEB_embeddings, spelling_error_list = saved_expensive_list

    for i, top_model in tqdm(enumerate(top_models), desc="Tabulating model scores...", ascii=False, ncols=75):
        os.chdir(cwd_path)
        
        [aug, model, timedir] = top_model[1].split('/')
        
        try: is_run_normal =  len(saved_df_full[i])==0
        except: is_run_normal = True
        
        if is_run_normal:
            gitdir = f'details_{aug}__{model}'

            try:    subprocess.run(['git', 'clone', f'https://huggingface.co/datasets/open-llm-leaderboard/{gitdir}'])
            except: continue

            try: 
                os.chdir(cwd_path+'/'+gitdir+'/'+timedir)
            except: 
                if timedir[-13]=='-' and timedir[-10]=='-':
                    time_separator = ':'
                if timedir[-13]==':' and timedir[-10]==':':
                    time_separator = '-'            
                try:
                    timedir = timedir[:-13] + time_separator + timedir[-12:-10] + time_separator + timedir[-9:]
                    os.chdir(cwd_path+'/'+gitdir+'/'+timedir)
                except: continue


            filter_key = filter_dict[benchmark_of_interest][0]
            if benchmark_of_interest=='mmlu': # Read and concatenate all 57 DataFrames into df_full
                pqt_filepath_error_flag, read_parquet_error_flag = False, False

                df_full = pd.DataFrame()
                for key in filter_key:
                    try:    pqt_filepath = subprocess.run(["find", ".", "-name", f"*{key}*"], capture_output=True, text=True).stdout.split('\n')[0].split('/')[1]
                    except: 
                        pqt_filepath_error_flag = True
                        break
                    pqt_filepath = cwd_path+'/'+gitdir+'/'+timedir+'/'+pqt_filepath
                    #print(f'pqt_filepath: {pqt_filepath}')

                    try: 
                        df_temp = pd.read_parquet(pqt_filepath)#.sample(frac=1, random_state=sample_random_state).reset_index(drop=True)
                        df_temp['mmlu_subject'] = key
                        df_temp['sampling_method'] = mmlu_subject_dict[key] if mmlu_subject_dict[key] in sampling_method_dict.keys() else list(sampling_method_dict.keys())[0]
                        df_full = pd.concat([df_full, df_temp], axis=0, ignore_index=True)
                        target_col = get_col_of_interest(df_full, filter_dict[benchmark_of_interest][1])
                        first_row_value = df_full[target_col]
                    except: 
                        problematic_model_pqt += [i]
                        read_parquet_error_flag = True
                        break
                if pqt_filepath_error_flag or read_parquet_error_flag:
                    continue

            else: # Just read 1 DataFrame for non-MMLU benchmarks
                try:    pqt_filepath = subprocess.run(["find", ".", "-name", f"*{filter_key}*"], capture_output=True, text=True).stdout.split('\n')[0].split('/')[1]
                except: continue
                pqt_filepath = cwd_path+'/'+gitdir+'/'+timedir+'/'+pqt_filepath
                #print(f'pqt_filepath: {pqt_filepath}')

                try: df_full = pd.read_parquet(pqt_filepath)#.sample(frac=1, random_state=sample_random_state).reset_index(drop=True)
                except: 
                    problematic_model_pqt += [i]
                    continue
            
        else:
            if i in problematic_model_pqt: continue
            df_full = saved_df_full[i]

        df_sample = pd.DataFrame()
        
        os.chdir(cwd_path)
        
        ##############################################################################
        ############# START : df_sample based on various sampling_method #############
        ##############################################################################
        if sampling_method=='random': 
            df_full = sampling_method_dict[sampling_method](df_full)
            df_sample = df_full.sample(frac=sample_pct, random_state=sample_random_state)
        
        elif 'clustering_' in sampling_method:
            if is_run_normal:
                if sampling_method=='clustering_Spectral_MTEB':
                    df_full, MTEB_embeddings, saved_expensive_list = sampling_method_dict[sampling_method](df_full, MTEB_embeddings, saved_expensive_list, sample_random_state)
                elif sampling_method=='clustering_Spectral_BERT':
                    df_full, bert_embeddings, saved_expensive_list = sampling_method_dict[sampling_method](df_full, bert_embeddings, saved_expensive_list, sample_random_state)
                else:
                    df_full = sampling_method_dict[sampling_method](df_full, sample_random_state)
            df_sample = df_full.groupby('Cluster', group_keys=False).apply(lambda x: x.sample(int(np.rint((sample_pct*len(df_full))*len(x)/len(df_full))), random_state=sample_random_state))#.sample(frac=1)
        
        elif 'quality_' in sampling_method:
            if is_run_normal:
                if sampling_method=='quality_spelling_error':
                    df_full, spelling_error_list, saved_expensive_list = sampling_method_dict[sampling_method](df_full, spelling_error_list, saved_expensive_list)
                else:
                    df_full = sampling_method_dict[sampling_method](df_full)
            quality_asc       = False if sampling_method[len('quality_'):] in ["compound_probability_distribution","avg_length_of_words","lexical_diversity"] else True
            df_sample = df_full.sort_values([sampling_method[len('quality_'):]], ascending=quality_asc, axis=0).head(int(df_full.shape[0]*sample_pct))
        
        elif 'difficult_' in sampling_method:
            if is_run_normal:
                df_full = sampling_method_dict[sampling_method](df_full)
            difficult_asc = True if sampling_method in ['difficult_Flesch_Readability'] else False
            df_sample = df_full.sort_values([sampling_method], ascending=difficult_asc, axis=0).head(int(df_full.shape[0]*sample_pct))

        elif sampling_method=='adaptive':
            if is_run_normal:
                df_full, bert_embeddings, MTEB_embeddings, spelling_error_list, saved_expensive_list = sampling_method_dict[sampling_method](df_full, bert_embeddings, MTEB_embeddings, spelling_error_list, saved_expensive_list, sample_random_state)
            df_sample = get_adaptive_df_sample(df_full, sample_pct, sample_random_state)

   
        ##############################################################################
        ############# FINISH: df_sample based on various sampling_method #############
        ##############################################################################
            
        for metric in metrics_dict.keys():
            target_col = get_col_of_interest(df_full, filter_dict[benchmark_of_interest][1])

            if target_col == "metrics":
                if type(df_full[target_col][0])==dict:
                    df_full[target_col] = df_full[target_col].apply(lambda val:val[f"{metric}"])
                    df_sample[target_col] = df_sample[target_col].apply(lambda val:val[f"{metric}"])
            
            if type(df_full[target_col][0])==str:
                df_full[target_col] = df_full[target_col].apply(convert_bool_to_int)
                df_sample[target_col] = df_sample[target_col].apply(convert_bool_to_int)
                
            val = [aug+'/'+model, float(f'{df_full[target_col].mean()*100:.2f}'), float(f'{df_sample[target_col].mean()*100:.2f}')]
            metrics_dict[metric].append(val)
        
        resume_df_full += [df_full]
        
        os.chdir(cwd_path)

    final_dict = {metric+'_'+metrics_dict[metric][i][0]:[metrics_dict[metric][i][1], metrics_dict[metric][i][2]] for metric in metrics_dict.keys() for i in range(len(metrics_dict[metric]))}
    final_df   = pd.DataFrame.from_dict(final_dict, orient='index', columns=['full_data', 'sample_data'])  
    
    return final_df, resume_df_full, problematic_model_pqt, saved_expensive_list


def get_ranking(df):
    df_cols = ['full_data', 'sample_data']
    ranking = []
    for col in df_cols:
        temp = [[df.index.tolist()[i], df[col].tolist()[i]] for i in range(df.shape[0])]
        temp = sorted(temp, key=lambda val: val[1], reverse=True)
        temp = {i+1: [temp[i][1], temp[i][0]] for i in range(len(temp))}
        ranking += [temp]
    return ranking[0], ranking[1]


def get_subset_ranking(benchmark_name, LLM_list, sampling_method, num_top_models=10, model_sampling_interval=1):
    top_models = get_top_models(num_top_models=num_top_models, benchmark_of_interest=benchmark_name, LLM_list=LLM_list, model_sampling_interval=model_sampling_interval)
    print()
    output_df, saved_df_full, problematic_model_pqt, saved_expensive_list = tabulate_score(top_models[0], top_models[1], sampling_method=sampling_method)
    return get_ranking(output_df)


def calculate_pearson_coefficient(full_data, subset_data):
    return pearsonr(full_data, subset_data).statistic


def calculate_wasserstein_distance(full_data, subset_data):
    return wasserstein_distance(full_data, subset_data)


def calculate_kl_divergence(full_data, subset_data):
    return sum(kl_div(full_data, subset_data))


def calculate_ks_test(full_data, subset_data):
    return ks_2samp(full_data, subset_data).statistic


def calculate_similarity_measures(full_data, subset_data, similarity_measure_keyword):
    ans = -1.0
    if similarity_measure_keyword=='Pearson Coefficient':
        ans = calculate_pearson_coefficient(full_data, subset_data)
    elif similarity_measure_keyword=='Wasserstein Distance':
        ans = calculate_wasserstein_distance(full_data, subset_data)
    elif similarity_measure_keyword=='KL Divergence':
        ans = calculate_kl_divergence(full_data, subset_data)
    elif similarity_measure_keyword=='Kolmogorov-Smirnov Test':
        ans = calculate_ks_test(full_data, subset_data)
    return ans


def computation_for_visualization(benchmark_name, LLM_list, num_top_models=10, model_sampling_interval=1, initial_sample_pct=.05, sample_pct_increment=.05, force_run_from_scratch=False):
    npy_filepath = f'{cwd_path}/npy_file/npy_{benchmark_name}_{num_top_models}model_{model_sampling_interval}modelsamplinginterval_{initial_sample_pct}initialpct_{sample_pct_increment}pctincrement.npy'
    if os.path.exists(npy_filepath)==True and force_run_from_scratch==False:
        if benchmark_name=='mmlu' and 'adaptive' not in sampling_method_dict.keys(): sampling_method_dict.update({'adaptive':calc_adaptive})
        if benchmark_name!='mmlu' and 'adaptive'     in sampling_method_dict.keys(): sampling_method_dict.pop('adaptive')
        sampling_method_result_dict = dict(np.load(npy_filepath, allow_pickle=True).item())
        num_iteration = np.floor((1.0-initial_sample_pct)/sample_pct_increment+.5).astype(int)+1
        sample_pct_list  = [initial_sample_pct + sample_pct_increment*i for i in range(num_iteration)]
        sampling_method_df_full_dict = {}
        return sampling_method_result_dict, sample_pct_list, sampling_method_df_full_dict

    assert initial_sample_pct > 0.0, f'{initial_sample_pct} cannot be less than equal to zero.'

    top_models = get_top_models(num_top_models=num_top_models, benchmark_of_interest=benchmark_name, LLM_list=LLM_list, model_sampling_interval=model_sampling_interval)
    print()

    num_iteration = np.floor((1.0-initial_sample_pct)/sample_pct_increment+.5).astype(int)+1

    sample_pct_list  = [initial_sample_pct + sample_pct_increment*i for i in range(num_iteration)]
    
    sampling_method_result_dict = defaultdict(list)
    
    sampling_method_df_full_dict = defaultdict(list)
    
    for sampling_method in sampling_method_dict.keys():
        final_rank_list  = []
        final_score_list = []
        
        saved_df_full  = [pd.DataFrame()]*len(top_models[0])
        problematic_model_pqt = []
        saved_expensive_list = [[], [], []]

        for i in range(num_iteration):
            sample_pct = initial_sample_pct + sample_pct_increment*i
            print(f'\nsampling_method:{sampling_method}, i:{i}, sample_pct:{sample_pct:.4f}')
            output_df, saved_df_full, problematic_model_pqt, saved_expensive_list = tabulate_score(top_models[0], top_models[1], sampling_method=sampling_method, sample_pct=sample_pct, saved_df_full=saved_df_full, problematic_model_pqt=problematic_model_pqt, saved_expensive_list=saved_expensive_list)
            full_ranking, subset_ranking = get_ranking(output_df)

            if i==0: sampling_method_df_full_dict[sampling_method] = saved_df_full
            
            model_rank_score_dict = defaultdict(list)
            for i, key in enumerate(full_ranking.keys()):
                model_rank_score_dict[full_ranking[key][1]] += [key, full_ranking[key][0]]
            for i, key in enumerate(subset_ranking.keys()):
                model_rank_score_dict[subset_ranking[key][1]] += [key, subset_ranking[key][0]]

            full_rank_list, subset_rank_list  = [], []
            full_score_list,subset_score_list = [], []

            for key in model_rank_score_dict.keys():
                full_rank_list    += [model_rank_score_dict[key][0]]
                full_score_list   += [model_rank_score_dict[key][1]]
                subset_rank_list  += [model_rank_score_dict[key][2]]
                subset_score_list += [model_rank_score_dict[key][3]]
            #print("Sampling Method :",sampling_method,"\n","Full score list: ",full_score_list,"\n","Subset rank list :",subset_score_list)
            
            temp = []
            for measure in similarity_measures_list:
                temp += [calculate_similarity_measures(full_rank_list, subset_rank_list, similarity_measure_keyword=measure)]
            final_rank_list.append(temp)

            temp = []
            for measure in similarity_measures_list:
                temp += [calculate_similarity_measures(full_score_list, subset_score_list, similarity_measure_keyword=measure)]
            final_score_list.append(temp) 

        final_rank_list = np.array(final_rank_list)
        final_score_list = np.array(final_score_list)
        
        sampling_method_result_dict[sampling_method] = [final_rank_list, final_score_list]
    
    # dim1: sampling method
    # dim2: rank, score
    # dim3: sample_pct iteration
    # dim4: similarity measures

    if os.path.exists(cwd_path+'/npy_file')==False:
        joiner = '\ '
        os.system(f"mkdir {joiner.join(cwd_path.split(' ')) if ' ' in cwd_path else cwd_path}/npy_file")
    np.save(npy_filepath, np.array(sampling_method_result_dict))
    return sampling_method_result_dict, sample_pct_list, sampling_method_df_full_dict


def plot_graphs(visual_raw_data, sample_pct_list, benchmark_name, num_top_models, model_sampling_interval, initial_sample_pct, sample_pct_increment):
    def capitalize(x):
        return chr(ord(x[0])-32)+x[1:]

    # dim1: sampling method
    # dim2: rank, score
    # dim3: sample_pct iteration
    # dim4: similarity measures

    enable_xlim, enable_concise_plots = True, True
    x_len = 0.52
    assert sample_pct_list[0]-0.01  > -0.00001, "initial_pct_list is too small"
    assert sample_pct_list[0]+x_len <  1.00001, "initial_pct_list is too big"
    
    visual_raw_data_keys = list(visual_raw_data.keys())

    dimension1 = len(visual_raw_data_keys)
    dimension2 = list(set([len(visual_raw_data[key]      ) for key in visual_raw_data_keys]))[0]
    dimension3 = list(set([len(visual_raw_data[key][0]   ) for key in visual_raw_data_keys]))[0]
    dimension4 = list(set([len(visual_raw_data[key][0][0]) for key in visual_raw_data_keys]))[0]

    metric_names = ['rank', filter_dict[benchmark_name][1]]
    color_choices = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*2

    num_vertical_plots,  num_horizontal_plots = dimension4, dimension2
    if enable_concise_plots: 
        num_vertical_plots, num_horizontal_plots = 1, 2
    fig, axes = plt.subplots(num_vertical_plots, num_horizontal_plots, figsize=(15, 5*num_vertical_plots if enable_concise_plots else 10*num_horizontal_plots))
    subplot_itr = 1
    for similarity_iterator in range(dimension4): # similarity measures: Pearson Coefficient, KL Divergence, ...
        for metric_iterator in range(dimension2): # rank (P.S.: rank for best model = 1), score (i.e. mc2, acc, ...)
            if enable_concise_plots:
                do_plot_conditions = [
                                      similarity_measures_list[similarity_iterator]=='Pearson Coefficient'  and metric_names[metric_iterator]=='rank',
                                      similarity_measures_list[similarity_iterator]=='Wasserstein Distance' and metric_names[metric_iterator]==filter_dict[benchmark_name][1],
                                     ]
                if True not in do_plot_conditions: continue
            ax = plt.subplot(num_vertical_plots, num_horizontal_plots, subplot_itr) #similarity_iterator*dimension2 + metric_iterator+1)
            for sampling_method_iterator in range(dimension1): # random, topic, ...
                linestyle = '-'
                if 'quality_' in visual_raw_data_keys[sampling_method_iterator]:
                    linestyle = '--'
                if 'clustering_' in visual_raw_data_keys[sampling_method_iterator]:
                    linestyle = '-.'
                if 'difficult_' in visual_raw_data_keys[sampling_method_iterator]:
                    linestyle = ':'
                plt.plot(sample_pct_list, visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator].T[similarity_iterator], color=color_choices[sampling_method_iterator], label=list(sampling_method_dict.keys())[sampling_method_iterator], linestyle=linestyle)
                plt.xlabel('Sample Intervals')
                plt.ylabel(f'{similarity_measures_list[similarity_iterator]}')
                if enable_xlim:
                    plt.xlim(left=sample_pct_list[0]-0.01, right=sample_pct_list[0]+x_len)
                plt.xticks(sample_pct_list[:int(x_len*100):10], [f'{int(pct*100)-1}%' for pct in sample_pct_list[:int(x_len*100):10]])
                if list(sampling_method_dict.keys())[sampling_method_iterator]=='Pearson Coefficient':
                    plt.axhline(y = 3, color = 'b', linestyle = ':', label = "90% Pearson Coefficient") 
            plt.legend()
            plt.title(f"{'MMLU'+capitalize(benchmark_name)[13:] if 'hendrycksTest' in benchmark_name else capitalize(benchmark_name)} {capitalize(metric_names[metric_iterator])}: {similarity_measures_list[similarity_iterator]}")
            subplot_itr += 1

    if os.path.exists(cwd_path+'/visualization_results')==False:
        os.system(f'mkdir {cwd_path}/visualization_results')
    plt.savefig(f"{cwd_path}/visualization_results/{'xlim_' if enable_xlim else ''}{benchmark_name}_{num_top_models}model_{model_sampling_interval}modelsamplinginterval_{initial_sample_pct}initialpct_{sample_pct_increment}pctincrement.JPG")
    plt.savefig(f"{cwd_path}/visualization_results/{'xlim_' if enable_xlim else ''}{benchmark_name}_{num_top_models}model_{model_sampling_interval}modelsamplinginterval_{initial_sample_pct}initialpct_{sample_pct_increment}pctincrement.PDF")

    
def plot_variance_graph(visual_raw_data, sample_pct_list, benchmark_name, num_top_models, model_sampling_interval, initial_sample_pct, sample_pct_increment):
    def capitalize(x):
        return chr(ord(x[0])-32)+x[1:]
 
    # dim1: sampling method
    # dim2: rank, score
    # dim3: sample_pct iteration
    # dim4: similarity measures
 
    metric_list = ['Rank','Score']
    threshold_list = [[0.9,25,1,0.1],[0.9,5,1,0.1]]

    visual_raw_data_keys = list(visual_raw_data.keys())
 
    dimension1 = len(visual_raw_data_keys)
    dimension2 = list(set([len(visual_raw_data[key]      ) for key in visual_raw_data_keys]))[0]
    dimension3 = list(set([len(visual_raw_data[key][0]   ) for key in visual_raw_data_keys]))[0]
    dimension4 = list(set([len(visual_raw_data[key][0][0]) for key in visual_raw_data_keys]))[0]
 
    num_horizontal_plots = dimension2
    num_vertical_plots   = dimension4
 
    fig, axes = plt.subplots(num_vertical_plots, num_horizontal_plots, figsize=(20, 10*num_horizontal_plots))
    #metric_names = [capitalize('rank'), capitalize(filter_dict[benchmark_name][1])]
    color_choices = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*2
 
    result = {}
 
    for similarity_iterator in range(dimension4): # similarity measures: Pearson Coefficient, KL Divergence, ...
        similarity = {}
        for metric_iterator in range(dimension2): # rank (P.S.: rank for best model = 1), score (i.e. mc2, acc, ...)
            ax = plt.subplot(num_vertical_plots, num_horizontal_plots, similarity_iterator*dimension2 + metric_iterator+1)
            metric_method = {}
            for sampling_method_iterator in range(dimension1): # random, topic, ...
                if similarity_iterator == 0:
                    if max(visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator][:40,similarity_iterator]) < threshold_list[metric_iterator][similarity_iterator]:
                        continue
                else:
                    if min(visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator][:40,similarity_iterator]) > threshold_list[metric_iterator][similarity_iterator]:
                        continue
 
                linestyle = '-'
                if 'quality_' in visual_raw_data_keys[sampling_method_iterator]:
                    linestyle = '--'
                if 'clustering_' in visual_raw_data_keys[sampling_method_iterator]:
                    linestyle = '-.'
                if 'difficult_' in visual_raw_data_keys[sampling_method_iterator]:
                    linestyle = ':'
                var_value = []
                pearson_val = []
 
                #for sample in range(10,len(visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator]),10):
                for sample in range(10,40,5):
                    var_value.append(np.var(visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator][sample:sample+5,similarity_iterator]))
                    # pearson_val.append(np.mean(visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator][sample:sample+10,similarity_iterator]))
                metric_method.update({sampling_method_iterator:var_value})
                plt.plot(range(len(var_value)),var_value, color=color_choices[sampling_method_iterator], label=list(sampling_method_dict.keys())[sampling_method_iterator]+" Variance", linestyle=linestyle, marker='*')
                
                plt.xlabel("Sample Intervals")
                plt.ylabel("Variance")
                plt.title(f"Variance Plot with Sampling Methods for {similarity_measures_list[similarity_iterator]} for {metric_list[metric_iterator]} with Threshold {threshold_list[metric_iterator][similarity_iterator]}")
                #plt.xticks(range(len(var_value)), [f'{var}-{var+10}%' for var in range(10,len(visual_raw_data[visual_raw_data_keys[sampling_method_iterator]][metric_iterator]),10)])
                plt.xticks(range(len(var_value)), [f'{var}-{var+5}%' for var in range(10,40,5)])
                plt.legend()
            similarity.update({metric_list[metric_iterator]:metric_method})
        result.update({similarity_measures_list[similarity_iterator]:similarity})
    if os.path.exists(cwd_path+'/visualization_results')==False:
        os.system(f'mkdir {cwd_path}/visualization_results')
    plt.savefig(f"{cwd_path}/visualization_results/var_{benchmark_name}_{num_top_models}model_{model_sampling_interval}modelsamplinginterval_{initial_sample_pct}initialpct_{sample_pct_increment}pctincrement.JPG")
    plt.savefig(f"{cwd_path}/visualization_results/var_{benchmark_name}_{num_top_models}model_{model_sampling_interval}modelsamplinginterval_{initial_sample_pct}initialpct_{sample_pct_increment}pctincrement.PDF")
    return result



######################
##### SANDBOX ########
######################

# ## Short code to view list of sampled questions
# df_full = sampling_method_df_full_dict['clustering_MTEB'][0] #df_full from 1st model
# sample_pct = .1
# sample_random_state = 1
# df_sample = df_full.groupby('Cluster', group_keys=False).apply(lambda x: x.sample(int(np.rint((sample_pct*len(df_full))*len(x)/len(df_full))), random_state=sample_random_state))#.sample(frac=1, random_state=sample_random_state)
# df_sample[df_sample.Cluster==4].example
