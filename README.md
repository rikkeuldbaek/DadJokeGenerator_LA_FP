
# **Final Project - Dad Joke Generator using GPT-2 and RNNs**
## **Cultural Data Science - Language Analytics** 
#### Author: Rikke Uldbæk (202007501)
#### Date: 13th of April 2023

<br>

# **5.1 GitHub link**
The following link is a link to the GitHub repository of the self assigned final project in the course Language Analytics (F23.147201U021.A). Within the GitHub repository all necessary code are provided to reproduce the results of the project.

https://github.com/rikkeuldbaek/DadJokeGenerator_LA_FP

<br>

# **5.2 Description**
For this final project I will be generating dad jokes using two different methods of text generation, namely a simple RNN (recurrent neural network) and a pretrained gpt-2 model. Both models will be finetuned and trained on a *dad jokes dataset* in order to generate text of the like. In order to make the generated dad jokes from the two models comparable, I have chosen to generate two kinds of jokes from prefixes. These prefixes will indicate the start phrase of the joke, so that the jokes will have similar nature (similar one liner structure). The first kind of joke has the following prefix: "What do you call", while the second kind of joke has the following prefix: "Why did the". The quality of the generated dad jokes from both models will be evaluated in the results section. 


<br>

# **5.3 Data**

The *dad jokes dataset* is a web scraped collection of signature one liner dad jokes. The dataset consists of 743 dad jokes with a mean length of 75 words. The data is both available within this repository and via Kaggle, please see resources for further information. 

<br>

# **5.4 Methods**
### **GPT 2**
For this project I have used the transformer model GPT-2 which is pretrained on a very large corpus of English text data. The language model comes in different sizes: GPT-Small, GPT-Medium, GPT-Large, and GPT-XL, and due to computational limiations the smallest GPT model is used (GPT-Small with 124 million parameters). The GPT-2 model is trained to find the next word in sentences, and is thus ideal for the purpose of this final project. The gpt-2 model is available via ```HuggingFace```, please see resources for further information.

<br>


### **RNN**

the RNN is constructed using tools from ```Keras - TensorFlow```. 

<br>

# **5.5 Repository Structure**
The scripts require a certain folder structure, thus the table below presents the required folders and their description and content.

|Folder name|Description|Content|
|---|---|---|
|```src```|text generator scripts |```gpt2_joke_generator.py```, ```RNN_joke_generator.py```, ```data.py```|
|```data```|.csv file of dad jokes|```dad-a-base.csv```|
|```out```|saved .txt files of generated dad jokes|```dad_jokes_gpt2_prefix_what.txt```, ```dad_jokes_gpt2_prefix_why.txt```, ```dad_jokes_RNN_prefix_what.txt```, ```dad_jokes_RNN_prefix_why.txt```|


the ```data.py```script located in ```src``` mainly preprocesses the data for the RNN. The ```gpt2_joke_generator.py``` and ```RNN_joke_generator.py``` located in ```src``` generates dad jokes which are saved in the ```out``` folder.

<br>

# **5.6 Usage and Reproducibility**
## **5.6.1 Prerequisites** 
In order for the user to be able to run the code, please make sure to have bash and python 3 installed on the used device. The code has been written and tested with Python 3.9.2 on a Linux operating system. In order to run the provided code for this assignment, please follow the instructions below.

<br>

## **5.6.2 Setup Instructions** 
**1) Clone the repository**
```python
git clone https://github.com/rikkeuldbaek/DadJokeGenerator_LA_FP
 ```

 **2) Setup** <br>
Setup virtual environment (```LA_fp_env```) and install required packages.
```python
bash setup.sh
```

<br>

## **5.6.3 Run the script** 
In order to run the three emotion classification scripts, please run the following command in the terminal after setting up. 
Please note that the three scripts take quite some time to run. 
```python
bash run.sh
```


<br>


# **5.7 Results**

<br>

# **Resources**
[GPT-2 - Huggingface](https://huggingface.co/gpt2 )

[RNN - Keras](https://www.tensorflow.org/guide/keras/rnn )

[Dad Jokes Data](https://www.kaggle.com/datasets/aryashah2k/dad-a-base-of-jokes)





IDEA: 
generate dad jokes using gpt-2 
generate dad jokes using another pretrained model
why not make my own model? There's simply too few data points in the dataset, for the a model to ever perform decently. 




RNN: 
performs bad because its given more text with more stopwords. we used the same model forcommments on articles in assignment 3(less stopwords). So when the model trains on jokes (with a lot of stopwords) it finds the words closest to the prefix phrase, and these are often stopwords. Too little data for the model to learn enough and perform well...




pitfalls .. 
the gpt2 model does not have a fixed amount of words to generate just a maximum (not the case for RNN)
RNN gives 1 sample
GPT-2 gives 20 samples.


RESOURCES:
[GPT-2 - Huggingface](https://huggingface.co/gpt2 )

[RNN - Keras](https://www.tensorflow.org/guide/keras/rnn )

[Dad Jokes Data](https://www.kaggle.com/datasets/aryashah2k/dad-a-base-of-jokes)