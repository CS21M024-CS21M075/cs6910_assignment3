# Brief Introduction to CS6910 - Deep Learning Assignment 3

The Assignment consists of 4 Major Parts as follows:

1. Building and Training a RNN Based Vanilla Model using the Google Dakshina Dataset

2. Building and Training a RNN Based Attention Model using the same Dataset

3. Visualization of Connectivity in Inter/Intra Sequences.

4. Lyrics Generation using GPT2 given a specific string.

The link to the wandb report:

## 1.Building and Training a RNN Based Vanilla Model using the Google Dakshina Dataset:

### Files associated:

### Model:

The Simple RNN Based Model is made using Multiple Layers of Encoder and Multiple Layers of Decoder, along with that Dense Layer is being used in the Output. The output is basically the whole vocabulary of the Corpus.

### Dataset:

For the task we have used Google Dakshina Dataset. Link: https://github.com/google-research-datasets/dakshina

The Dataset consists of 12 Different Asian Languages and we can find the sequences of English to 12 Different Languages in the Dataset.

In this case study we are explicitely using "Hindi" Language (i.e. code = 'hi')

Hindi is consisting with 44204 Training Samples, 4358 Validation Samples and 4502 Test Samples.

### Training:

For Training Purposes we have used the "Bayesian Sweep" Functionality provided by WANDB and Tuned our HyperParametes based on the below options.

Sweep Configuration is as follows:

```
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "WordLevel_Validation_Accuracy",
  "goal": "maximize"
  },
  "parameters": {
        "hidden_layer_size": {
            "values": [128,256,512]
        },
        "num_encoder_layers": {
            "values": [1,2,3]
        },
         "num_decoder_layers": {
            "values": [1,2,3]
        },
        "learning_rate": {
            "values": [0.001,0.0001]
        },
        "optimizer": {
            "values": ['adam','rmsprop','nadam']
        },
        
        "batch_size": {
            "values": [64,128,256]
        },
        
        "model_name": {
            "values": ["RNN","GRU","LSTM"]
        },
 
        "embedding_size": {
            "values": [128,256,512]
        },
        
        "dropout": {
            "values": [0.1,0.2,0.3]
        },
                    
        "epochs": {
            "values": [20,25]
        },
      
        "beam_size": {
            "values": [0]
        },
        
        
    }
}
```

Based on the Training we have received the below best Hyper-Parameters:

Hidden Layer Size = 512

Number of Encoder Layers = 3, Number of Decoder Layers = 3,

Model Name = LSTM, Optimizer = NADAM, Learning Rate = 0.001,

Embedding Size = 512, Dropout = 0.3, Batch Size = 64,

Number of Epochs = 25, Beam Width = 0

Based on the defined model, 

Character wise Training Accuracy = **99.26%**

Character wise Validation Accuracy = **95.43%**

Word wise Validation Accuracy = **37.36%** 

### Testing:

The Best Model is then tested over the given Test Samples.

Word Wise Test Accuracy = **37.09%**

### Visualisation:

For this part we have visualized a Grid for some random 20 samples from the test dataset and the Input, Target and Predicted Sequences have been plotted.



## 2. Building and Training a RNN Based Attention Model using the Google Dakshina Dataset

### Files associated:

### Model:

The Attention Based RNN Based Model is made using Only Single Layer of Encoder and another one Layer of Decoder, along with that the Attention Layer and Dense Layer is being used in the Output. The output is basically the whole vocabulary of the Corpus.

### Dataset:

For the task we have used Google Dakshina Dataset. Link: https://github.com/google-research-datasets/dakshina

The Dataset consists of 12 Different Asian Languages and we can find the sequences of English to 12 Different Languages in the Dataset.

In this case study we are explicitely using "Hindi" Language (i.e. code = 'hi')

Hindi is consisting with 44204 Training Samples, 4358 Validation Samples and 4502 Test Samples.

### Training:

For Training Purposes we have used the "Bayesian Sweep" Functionality provided by WANDB and Tuned our HyperParametes based on the below options.

Sweep Configuration is as follows:

```
sweep_config = {
  "name": "Attention Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "Attention_Wordwise_Val_Accuracy",
  "goal": "maximize"
  },
  "parameters": {
        "hidden_layer_size": {
            "values": [128,256]
        },

        "learning_rate": {
            "values": [0.001]
        },
        "optimizer": {
            "values": ['adam','nadam']
        },
        
        "batch_size": {
            "values": [256, 512]
        },
        
        "model_name": {
            "values": ["LSTM"]
        },
 
        "embedding_size": {
            "values": [256, 512]
        },
        
        "dropout": {
            "values": [0.3]
        },
                    
        "epochs": {
            "values": [15,20,25]
        },
      
        
        
    }
}

```

Based on the Training we have received the below best Hyper-Parameters:

Hidden Layer Size = 384,

Number of Encoder Layers = 1, Number of Decoder Layers = 1,

Model Name = LSTM, Optimizer = ADAM, Learning Rate = 0.001,

Embedding Size = 512, Dropout = 0.3, Batch Size = 512,

Number of Epochs = 20

Based on the defined model, 

Character wise Training Accuracy = **97.26%**

Character wise Validation Accuracy = **95.93%**

Word wise Validation Accuracy = **41.35%** 

### Testing:

The Best Model is then tested over the given Test Samples.

Word Wise Test Accuracy = **41.02%**

### Visualisation:

There are two Visualizations in this part of the case study.

1. We have visualized a Grid for some random 20 samples from the test dataset and the Input, Target and Predicted Sequences have been plotted.

2. Attention Heatmaps are being visualized for 10 random samples from the test data.

## 3. Visualization of Connectivity in Inter/Intra Sequences:

## 4. Lyrics Generation using GPT2 given a specific string:

In this part we had to fine tune the GPT2 model to generate the lyrics of english songs.

### Dataset
For fine tuning we have used the dataset2(https://www.kaggle.com/paultimothymooney/poetry) provided in the question. 

The dataset consists of the multiple text files where each text file contained all the songs of a particular artist.
We have merged all the files provided in dataset2 into one single text file named "merged_file.txt".

We have uploaded the merged_file.txt and code for fine tuning GPT-2 into the folder "Question 8 GPT-2".

### Running the code:-
For Running the code you can run all the cells of the notebook line by line.
The GPT-2 model will get fine tuned for generating the lyrics of the english songs.
However, make sure you have entered the file location of merged_file.txt properly and file location for creating new files for train.txt and eval.txt

### Conclusion:-
Using this we were able to fine tune the GPT-2 and generate the lyrics for the engisg songs
