### Final Project - Dad Jokes Generator
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 11th of May 2023

#--------------------------------------------------------#
############## DAD JOKES GENERATOR - GPT-2 ###############
#--------------------------------------------------------#
 
# (please note that some of this code has been adapted from https://www.kaggle.com/code/aryashah2k/gpt-2-dad-joke-generator/notebook)

# Install packages
from transformers import pipeline
import pandas as pd
import os
import tensorflow as tf
import gpt_2_simple as gpt2



########################## DATA ##########################
file = os.path.join(os.getcwd(), "data", "dad-a-base.csv")


########################## DOWNLOAD MODEL ##########################
# download the smallest gpt-2 model with 124 million parameters
gpt2.download_gpt2(model_name="124M")

# initialize model
sess = gpt2.start_tf_sess()


#--------------------------------------------------------#
############## "WHAT DO YOU CALL.." JOKES ################
#--------------------------------------------------------#

############## FINETUNE GPT-2 ON DAD JOKES ################
gpt2.finetune(sess,
              dataset=file,
              model_name='124M', # smallest GPT-2 model
              steps=300, # number of steps to train (smaller steps is better for short text)
              restore_from='fresh', # training from base GPT-2
              run_name='dadjokes_gpt2_prefix_what', # folder name for saving model and checkpoint
              print_every=50, # print every n steps in training process
              sample_every=10, #prints n examples for every printed step
              )


######### GENERATE NEW DAD JOKES USING GPT-2 - PREFIX ##########
generated_file_what = 'out/dad_jokes_gpt2_prefix_what1.txt'
input_prompt_what = "What do you call"

gpt2.generate_to_file(sess, run_name = 'dadjokes_gpt2_prefix_what',
                      destination_path=generated_file_what , #filename of generated text
                      length=75, # max length of generated text
                      temperature= 1, #the larger value the more original txt is generated. (default = 0.7)
                      nsamples=20, # n text prompts generated
                      prefix=input_prompt_what, # indicator of starting text for GPT-2
                      truncate="<|endoftext|>", # indicator of ending text for GPT-2
                      include_prefix=True, #remove prefix 
                      sample_delim=''
                      )



#--------------------------------------------------------#
################ "WHY DID THE.. " JOKES ##################
#--------------------------------------------------------#

############## FINETUNE GPT-2 ON DAD JOKES ################
# initialize model
sess1 = gpt2.start_tf_sess()

gpt2.finetune(sess1,
              dataset=file,
              model_name='124M', # smallest GPT-2 model
              steps=300, # number of steps to train (smaller steps is better for short text)
              restore_from='fresh', # training from base GPT-2
              run_name='dadjokes_gpt2_prefix_why', # folder name for saving model and checkpoint
              print_every=50, # print every n steps in training process
              sample_every=10, #prints n examples for every printed step
              )


######### GENERATE NEW DAD JOKES USING GPT-2 - PREFIX ##########
generated_file_why = 'out/dad_jokes_gpt2_prefix_what1.txt'
input_prompt_why = "Why did the"

gpt2.generate_to_file(sess, run_name = 'dadjokes_gpt2_prefix_why',
                      destination_path=generated_file_why , #filename of generated text
                      length=75, # max length of generated text
                      temperature= 1, #the larger value the more original txt is generated. (default = 0.7)
                      nsamples=20, # n text prompts generated
                      prefix=input_prompt_why, # indicator of starting text for GPT-2
                      truncate="<|endoftext|>", # indicator of ending text for GPT-2
                      include_prefix=True, # include prefix
                      sample_delim=''
                      )