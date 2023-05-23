# **Final Project - Generating Dad Jokes using GPT-2 and RNNs**
## **Cultural Data Science - Language Analytics** 
#### Author: Rikke Uldbæk (202007501)
#### Date: 11th of May 2023

<br>

# **5.1 GitHub link**
The following link is a link to the GitHub repository of the self assigned final project in the course Language Analytics (F23.147201U021.A). Within the GitHub repository all necessary code are provided to reproduce the results of the project.

https://github.com/rikkeuldbaek/DadJokeGenerator_LA_FP

<br>

# **5.2 Description**
For this final project I will be generating dad jokes using two different methods of text generation, namely a simple RNN (recurrent neural network) and a pretrained GPT-2 model. Both models will be finetuned and trained on a *dad jokes dataset* in order to generate text of the like. In order to make the generated dad jokes from the two models comparable, I have chosen to generate two kinds of jokes from prefixes. These prefixes will indicate the start phrase of the joke, so that the jokes will have similar nature (similar one liner structure). The first kind of joke has the following prefix: "What do you call", while the second kind of joke has the following prefix: "Why did the". The quality of the generated dad jokes from both models will be evaluated in the results section. 


<br>

# **5.3 Data**
The *dad jokes dataset* is a web scraped collection of signature one liner dad jokes. The dataset consists of 743 dad jokes with a mean length of 75 words. The data is both available within this repository and via Kaggle, please see resources for further information. 

<br>

# **5.4 Methods**
### **GPT 2**
For this project I have used the transformer model GPT-2 which has been pretrained on a very large corpus of English text data. The language model comes in different sizes: GPT-Small, GPT-Medium, GPT-Large, and GPT-XL, and due to computational limiations the smallest GPT model is used (GPT-Small with 124 million parameters). The GPT-2 model is trained to find the next word in sentences, thus being ideal for the purpose of this final project. The GPT-2 model is available via ```HuggingFace```, please see resources for further information.

<br>

### **RNN**
Similarly I will be using a recurrent neural network (RNN) to generate dad jokes. RNNs are a type of sophisticated neural networks where the connection between the nodes are circular, thus being able to account for the sequential and temporal aspects of language. However, simple RNNs have a hard time dealing with long distance dependencies which are found in the nature of language, i.e., words from long ago have less impact than words from recent time (also known as the vanishing gradient problem). To overcome this problem, I have introduced a Gated Recurrent Unit layer (GRU) with 350 units in my RNN model. GRUs are an improved version of the simple RNNs, as they are able avoid phasing out the information through time, thus storing relevant information from long time ago. The RNN model of this project is constructed using tools from ```Keras - TensorFlow```, please see resources for further information. 

<br>

# **5.5 Repository Structure**
The scripts require a certain folder structure, thus the tables below presents the required folders and their description and content.

|Folder name|Description|Content|
|---|---|---|
|```src```|dad joke generator scripts and data preprocessing script|```gpt2_joke_generator_what.py```,```gpt2_joke_generator_why.py```, ```RNN_joke_generator_what.py```, ```RNN_joke_generator_why.py```, ```data.py```|
|```data```|.csv file of dad jokes|```dad-a-base.csv```|
|```out```|saved .txt files of generated dad jokes|```dad_jokes_gpt2_what.txt```, ```dad_jokes_gpt2_why.txt```, ```dad_jokes_RNN_what.txt```, ```dad_jokes_RNN_why.txt```|


The ```data.py```script located in ```src``` only preprocesses the data for the RNN. The ```gpt2_joke_generator_what.py```, ```gpt2_joke_generator_why.py```, ```RNN_joke_generator_what.py```, and ```RNN_joke_generator_why.py``` located in ```src``` generates dad jokes which are saved in the ```out``` folder.

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
In order to run the GPT-2 and RNN model, please run the following command in the terminal after setting up. 
Please note that the scripts take quite some time to run since two types of jokes are generated from each model.
```python
bash run.sh
```

<br>


## **5.6.4 Script arguments**
The GPT-2 and RNN model have the following default arguments stated in the table below. These arguments can be modified and adjusted in the ```run.sh``` script. If no modifications are added, default parameters are run. In case help is needed, please write ```--help``` in continuation of the code below instead of writing an argument.

```python
#GPT-2 model
python src/gpt2_joke_generator_what.py #add arguments here or --help
python src/gpt2_joke_generator_why.py #add arguments here or --help

#RNN model
python src/RNN_joke_generator_what.py #add arguments here or --help
python src/RNN_joke_generator_why.py #add arguments here or --help
```

<br>

The following arguments are available for the ```data.py```  script. These can be adjusted if the user wants to run the models on new data with a different folder structure.

|Argument|Type|Default|
|---|---|---|
|--folder|string|"data"|
|--file|string|"dad-a-base.csv"|



<br>

The following arguments are available for the ```gpt2_joke_generator_what.py``` and  ```gpt2_joke_generator_why.py``` scripts. Note that the ```--prefix_what``` argument is reserved to the ```gpt2_joke_generator_what.py``` script, and the ```--prefix_why``` argument is reserved to the ```gpt2_joke_generator_why.py``` script.

|Argument|Type|Default|
|---|---|---|
|--path_to_data|string|os.path.join(os.getcwd(), "data")|
|--file|string|"dad-a-base.csv"|
|--steps|integer|300|
|--restore_from|string|"fresh"|
|--prefix_what|string|"What do you call"|
|--prefix_why|string|"Why did the"|
|--gen_text_len|integer|75|
|--temperature|float|1.0|


<br>

The following arguments are available for the ```RNN_joke_generator_what.py``` and ```RNN_joke_generator_why.py``` scripts. Note that the ```--prefix_what``` argument is reserved to the ```RNN_joke_generator_what.py``` script, and the ```--prefix_why``` argument is reserved to the ```RNN_joke_generator_why.py``` script.

|Argument|Type|Default|
|---|---|---|
|--n_epochs|integer|30|
|--batch_size|integer|50|
|--verbose|integer|1|
|--prefix_what|string|"What do you call"|
|--prefix_why|string|"Why did the"|
|--n_next_words|integer|12|


### **Important to note** <br>
The ```data.py``` is automatically called upon when running the RNN model scripts through the ```run.sh``` script, thus the arguments for ```data.py``` must be parsed to the RNN model scripts inside the ```run.sh``` bash file:

````python 
#RNN model
python src/RNN_joke_generator_what.py --arguments_for_model --arguments_for_data
python src/RNN_joke_generator_why.py --arguments_for_model --arguments_for_data
````


# **5.7 Results**

## **5.7.1 Results of the GPT-2 dad joke generator**
The results of the GPT-2 model looks fairly fine. The model has been specified to generate 15 different jokes of each kind ("What do you call" and "Why did the"), thus multiple jokes were evaluated. The models seems to be picking up on the signature one liner dad joke style, with a sentence structure using a question and an answer. However, the majority of the generated jokes are lacking sematic quality or humor:

<br>

>_What do you call someone with no nose? Someone without a nose?_

>_What do you call a gorilla by his fierce little teeth? A gummy beast!_

>_Why did the barrel go to the bathroom? He was going to need a space bar._

>_Why did the spider say stay still?  So soft it wondered if it was silk or silkect._

<br>

Although the majority of GPT-2 generated dad jokes are pretty poor, some of the jokes are actually funny. However, it turns out that these jokes are not always original. Using a simple google search to check whether or not the dad jokes are already present online, I found that most of the generated jokes are not original, i.e., they are already present somehwere on the internet. This is most likely due to the fact that GPT-2 is pretrained on web-data, and thus are likely to rely on it's pre-existing knowledge of available dad jokes. The ```temperature``` argument, specifying originality of generated text, could be tweaked to overcome the problem, however this is a trade-off between originality and semantic quality of the sentence. For instance the GPT-2 model generated the following unoriginal jokes (pre-existing dad jokes found online): <br>

<br>

>_What do you call a pig that knows karate? A pork chop!_ 	

>_Why did the barber win the race? He took a short cut._

<br>

Nonetheless, the finetuned GPT-2 model did actually create one new original "What do you call"-dad joke. Using a little google search, I was not able to find a similar joke, hence I consider the following joke a new original dad joke:

<br>

>_What do you call a troublesome dog? A Terrier-izer._

<br>

Please navigate to the ```out``` folder to inspect all GPT-2 generated dad jokes.

<br>

## **5.7.2 Results of the RNN dad joke generator**
The results of the RNN model are quite poor. The model can only generate one joke of each kind ("What do you call" and "Why did the"), hence only two generated jokes are evaluated. From the generated jokes below it is very obvious that the jokes lack grammatical and semantic quality. 

<br>

> _What Do You Call A Cow With A Trampoline A Milk Shake A Fish A Rooster_

> _Why Did The Coffee Fail The Driving Test It Never Came To A Full Stop_ 

<br>

The poor performance is not surprising, since the model only has been trained on a very little text corpus of 743 relatively short sentences (dad jokes). There's simply too little data in the dataset, and this is definitely reflected in the quality of generated dad jokes. However, both dad jokes above have similar structures of pre-existing jokes which are found in the *dad jokes dataset*: "What do you call a cow on a trampoline? A milkshake." and "Why did the sentence fail the driving test? It never came to a full stop". Hence the RNN model almost replicates pre-existing jokes from the *dad jokes dataset*, making it pretty poor model for generating new text. 

<br>

Please navigate to the ```out``` folder to inspect all RNN generated dad jokes.

<br>

Overall, the finetuned GPT-2 model outperforms the finetuned RNN when generating dad jokes. This makes pretty good sense as the GPT-2 model is pertrained on a very large english corpus, and thus have learned an inner representation of the English language beforehand. Contraily the simple RNN is only trained on this very little English corpus without any pre-existing knowledge of English language. After all, one could argue that it is not fair to compare the performance of a finetuned GPT-2 model and a finetuned RNN, as these models are indeed very different in their nature. 

<br>


# **Resources**
[GPT-2 - Huggingface](https://huggingface.co/gpt2 )

[RNN - Keras](https://www.tensorflow.org/guide/keras/rnn )

[GRU - Keras](https://keras.io/api/layers/recurrent_layers/gru/)

[Dad Jokes Data](https://www.kaggle.com/datasets/aryashah2k/dad-a-base-of-jokes)
