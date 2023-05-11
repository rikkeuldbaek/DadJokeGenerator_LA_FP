### Final Project - Emotion Classification on Reddit posts
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 19th of April 2023

#--------------------------------------------------------#
################# EMOTION CLASSIFICATION #################
#--------------------------------------------------------#

# Install packages
from transformers import pipeline
import pandas as pd
import os
import tensorflow as tf
import gpt_2_simple as gpt2



########################## DATA ##########################
file = os.path.join(os.getcwd(), "data", "dad-a-base.csv")
#data = pd.read_csv(file, index_col=0)




