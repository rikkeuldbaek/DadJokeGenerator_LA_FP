### Final Project - Dad Jokes Generator
## Cultural Data Science - Language Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 11th of May 2023

#--------------------------------------------------------#
################## DAD JOKES GENERATOR ###################
#--------------------------------------------------------#

# Install packages
import pandas as pd
import os

########################## DATA ##########################
file = os.path.join(os.getcwd(), "data", "dad-a-base.csv")
data = pd.read_csv(file)

################## DATA INSPECTION #######################
# mean length of jokes 
mean_length =data['Joke'].str.len().mean()
print("Mean length of jokes in the data : ",round(mean_length, 0)) 

