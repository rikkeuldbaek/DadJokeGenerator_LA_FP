### Final Project - Dad Jokes Generator
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 11th of May 2023

#--------------------------------------------------------#
############### DAD JOKES GENERATOR - RNN ################
#--------------------------------------------------------#
 
# (please note that some of this code has been adapted from class sessions)


# data processing tools 
import string, os, sys
import pandas as pd
import numpy as np

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)

# Save things
from joblib import dump
import pickle

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

    #add arguments for data preprocessing
    parser.add_argument("--folder", type=str, default= "data", help= "Specify folder where .csv file is located.") 
    parser.add_argument("--file", type=str, default= "dad-a-base.csv", help= "Specify filename of .csv.") 

    #add arguments for training the model
    parser.add_argument("--n_epochs", type=int, default= 30, help= "Specify number of epochs. More epochs increase accuracy but also computational time of running.") 
    parser.add_argument("--batch_size", type=int, default= 50, help= "Specify size of batch size. The batch size refers to the number of samples which are propagated through the network.")
    parser.add_argument("--verbose", type=int, default= 1, help= "Specify whether the training progress for each epoch should be displayed.") 
    
    #add arguments for running the model
    parser.add_argument("--prefix_why", type=str, default= "why did the", help= "Specify prefix for text generation.") 
    parser.add_argument("--n_next_words", type=int, default= 12, help= "Specify number of next words following the prefix.")


    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments


################# IMPORT DATA ##############
import data as dt

args = input_parse()
data = dt.data_load(args.folder, args.file)
joke_corpus = dt.data_clean(data)
tokenizer, total_words = dt.data_tokenize(joke_corpus)
predictors, label, max_sequence_len, total_words, tokenizer = dt.data_seq_and_pad(tokenizer, total_words, joke_corpus)

##################### TRAIN THE MODEL #####################

# Create model
def model_func(max_sequence_len, total_words, predictors, label, n_epochs, batch_size, verbose):
    # Contruct model
    model = hf.create_model(max_sequence_len, total_words) 

    # Create history
    history = model.fit(predictors,
                        label, 
                        epochs= n_epochs, 
                        batch_size= batch_size, 
                        verbose=1) 
    

    return history, model


############# SAVE THE MODEL, TOKENIZER & MAX_SEQ_LENGTH #############
def save_func(history, model, tokenizer, max_sequence_len):
    file_path = os.path.join(os.getcwd(),"models_RNN") 
    tf.keras.models.save_model(
        model,
        file_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True
    ) 
    
    #save tokenizer
    outpath_tokenizer = os.path.join(os.getcwd(), "models_RNN", "tokenizer.pickle")
    dump(tokenizer, open(outpath_tokenizer, 'wb'))
    
    #save max_sequence_len
    f = open("models_RNN/max_sequence_len.txt", "w")
    f.write(str(max_sequence_len))
    f.close()

    return()

#################### RUN THE MODEL #######################
def run_model(prefix_why, n_next_words, max_sequence_len, tokenizer):
    #load model
    file_path = os.path.join(os.getcwd(),"models_RNN")
    loaded_model = tf.keras.models.load_model(file_path) 
    generated_text_why = hf.generate_text(prefix_why, n_next_words, loaded_model, max_sequence_len, tokenizer) # Generated text 
    
    print(generated_text_why)

    #save generated text (why joke)
    f = open("out/dad_jokes_RNN_why.txt", "w")
    f.write(str(generated_text_why))
    f.close()

    return()



#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    
    #train the model
    print("Training the RNN text generator on dad-jokes..")
    history, model = model_func(max_sequence_len, total_words, predictors, label, args.n_epochs, args.batch_size, args.verbose)
    print("Training done!")
    save_func(history, model, tokenizer, max_sequence_len)
    print("Saving model and metrics..")

    #run the model
    print("Running the model..")
    run_model(args.prefix_why, args.n_next_words, max_sequence_len, tokenizer)


if __name__ == '__main__':
    main()