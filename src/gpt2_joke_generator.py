### Final Project - Dad Jokes Generator
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 11th of May 2023

#--------------------------------------------------------#
############## DAD JOKES GENERATOR - GPT-2 ###############
#--------------------------------------------------------#

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

########################## FINETUNE GPT-2 ON DAD JOKES ##########################
sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file,
              model_name='124M', # smallest GPT-2 model
              steps=150, # number of steps to train (smaller steps is better for short text)
              restore_from='fresh', # training from base GPT-2
              run_name='dadjokes_gpt2_prefix', # folder name for saving model and checkpoint
              print_every=50, # print every n steps in training process
              sample_every=10, #prints n examples for every printed step
              )


################### GENERATE NEW DAD JOKES USING GPT-2 - PREFIX ####################
generated_file = 'out/dad_jokes_gpt2_prefix.txt'
input_prompt = "Why did the chicken cross the road?"

gpt2.generate_to_file(sess, run_name = 'dadjokes_gpt2_prefix',
                      destination_path=generated_file , #filename of generated text
                      length=75, # max length of generated text
                      temperature= 1, #the larger value the more original txt is generated. (default = 0.7)
                      nsamples=15, # n text prompts generated
                      batch_size=25, 
                      prefix=input_prompt, # indicator of starting text for GPT-2
                      #truncate="<|endoftext|>", # indicator of ending text for GPT-2
                      #include_prefix=False, #remove prefix 
                      sample_delim=''
                      )
