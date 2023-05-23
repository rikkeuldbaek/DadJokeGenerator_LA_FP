### Final Project - Dad Jokes Generator
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 11th of May 2023

#--------------------------------------------------------#
############## DAD JOKES GENERATOR - GPT-2 ###############
############## "WHAT DO YOU CALL.." JOKES ################
#--------------------------------------------------------#
  
# (please note that some of this code has been adapted from https://www.kaggle.com/code/aryashah2k/gpt-2-dad-joke-generator/notebook)

# Install packages
from transformers import pipeline
import pandas as pd
import os
import tensorflow as tf
import gpt_2_simple as gpt2

# Scripting
import argparse

# fetch data
import data as dt


###################### PARSER #########################
def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()

    #arguments for data preprocessing
    parser.add_argument("--folder", type=str, default= "data", help= "Specify folder where .csv file is located.") 
    parser.add_argument("--file", type=str, default= "dad-a-base.csv", help= "Specify filename of .csv.") 

    #arguments for gpt2
    parser.add_argument("--steps", type=int, default= 300, help= "Specify number of steps to train.") 
    parser.add_argument("--restore_from", type=str, default= "fresh", help= "Specify to restore from base gpt-2.") 
    parser.add_argument("--prefix_what", type=str, default= "What do you call", help= "Specify input prompt for model.") 
    parser.add_argument("--gen_text_len", type=int, default= 75, help= "Specify max length of generated text.") 
    parser.add_argument("--temperature", type=float, default= 1.0, help= "Specify originality of generated text, the larger value the more original txt is generated (default = 0.7).") 

    # parse the arguments from the command line 
    args = parser.parse_args()
    
    #define a return value
    return args #returning arguments


################# IMPORT DATA ##############
args = input_parse()
data = dt.data_load(args.folder, args.file)


########################## DOWNLOAD MODEL ##########################
def download_and_initialize_model():
    # download the smallest gpt-2 model with 124 million parameters
    gpt2.download_gpt2(model_name="124M")

    # initialize model
    sess = gpt2.start_tf_sess()
    return(gpt2, sess)


############## FINETUNE GPT-2 ON DAD JOKES ################
def finetune_model(gpt2, sess, data, steps, restore_from):
    gpt2.finetune(sess,
                dataset=data,
                model_name='124M', # smallest GPT-2 model
                steps=steps, # number of steps to train (smaller steps is better for short text)
                restore_from=restore_from, # training from base GPT-2
                run_name='dadjokes_gpt2_prefix_what', # folder name for saving model and checkpoint
                print_every=50, # print every n steps in training process
                sample_every=10, #prints n examples for every printed step
                )
    return(gpt2, sess)


######### GENERATE NEW DAD JOKES USING GPT-2 - PREFIX ##########

def joke_generator(gpt2, sess, prefix_what, gen_text_len, temperature):

    #predefined variables 
    generated_file = 'out/dad_jokes_gpt2_what1.txt'

    gpt2.generate_to_file(sess, run_name = 'dadjokes_gpt2_prefix_what',
                        destination_path=generated_file , #filename of generated text
                        length= gen_text_len, # max length of generated text
                        temperature= temperature, #the larger value the more original txt is generated. (default = 0.7)
                        nsamples=15, # n text prompts generated
                        prefix=prefix_what, # indicator of starting text for GPT-2
                        truncate="<|endoftext|>", # indicator of ending text for GPT-2
                        include_prefix=True, #remove prefix 
                        sample_delim=''
                        )
    return()

#################### MAIN FUNCTION #######################
def main():
    args = input_parse()
    
    # gpt-2 model
    gpt2, sess = download_and_initialize_model()
    gpt2, sess = finetune_model(gpt2, sess, data, args.steps, args.restore_from)
    joke_generator(gpt2, sess, args.prefix_what, args.gen_text_len, args.temperature)


if __name__ == '__main__':
    main()
