### Final Project - Dad Jokes Generator
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 11th of May 2023

#--------------------------------------------------------#
############ DATA PREPROCESSING - DAD JOKES ##############
#--------------------------------------------------------#

# data processing tools
import string, os, sys
import pandas as pd
import numpy as np
np.random.seed(42)
#from random import sample
import random

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.preprocessing.text import Tokenizer

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import predefined functions 
sys.path.append(os.path.join("utils"))
import helper_func as hf

# Scripting
import argparse

###################### PARSER ############################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #add arguments for data.py
    parser.add_argument("--folder", type=str, default= "data", help= "Specify folder where .csv file is located.") 
    parser.add_argument("--file", type=str, default= "dad-a-base.csv", help= "Specify filename of .csv.") 


    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments



#################### DATA PREPROCESSING ####################

# Load dad-a-base .csv file 
def data_load(folder, file):
    file_path = os.path.join(os.getcwd(), folder, file)
    data = pd.read_csv(file_path)
    return data

# Clean data
def data_clean(data):

    joke_corpus = [hf.clean_text(x) for x in data["Joke"]] # clean the text corpora 
    return joke_corpus


# Tokenize the data
def data_tokenize(joke_corpus):
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(joke_corpus)
    total_words = len(tokenizer.word_index) + 1
    return tokenizer, total_words


## Get sequence of tokens and pad the sequences
def data_seq_and_pad(tokenizer, total_words, joke_corpus):
    # Create input sequences of tokens
    inp_sequences = hf.get_sequence_of_tokens(tokenizer, joke_corpus)
    # Overcome fixed dimensionality and pad the sequences with 0's until of same length
    predictors, label, max_sequence_len = hf.generate_padded_sequences(inp_sequences, total_words) 
    return predictors, label, max_sequence_len, total_words, tokenizer 


#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    print("Initializing data preprocessing of dad jokes..")
    data = data_load(args.folder, args.file)
    joke_corpus = data_clean(data)
    tokenizer, total_words = data_tokenize(joke_corpus)
    predictors, label, max_sequence_len, total_words, tokenizer = data_seq_and_pad(tokenizer, total_words, joke_corpus)
    print("Data preprocessing done!")

if __name__ == '__main__':
    main()
