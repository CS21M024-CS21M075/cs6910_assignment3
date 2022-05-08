# Brief Introduction to CS6910 - Deep Learning Assignment 3

### Submitted by CS21M024: Joyojyoti Acharya, CS21M075: Vrushab Ramesh Karia

The Assignment consists of 4 Major Parts as follows:

1. Building and Training a RNN Based Vanilla Model using the Google Dakshina Dataset

2. Building and Training a RNN Based Attention Model using the same Dataset

3. Visualization of Connectivity in Inter/Intra Sequences.

4. Lyrics Generation using GPT2 given a specific string.

The link to the wandb report: https://wandb.ai/cs21m024_cs21m075/CS6910-Assignment-3/reports/-CS6910-Deep-Learning-Assignment-3--VmlldzoxODc4NjQ1?accessToken=m9iw3r5s7u6qzg5m6eij9sqt84l1r3712qxbiqsos0b09uf22qxu7igcrb6uopkl

## 1.Building and Training a RNN Based Vanilla Model using the Google Dakshina Dataset:

### Files associated:

1. DL_assignment3.ipynb                                 ~ Basic code structure of Encoder-Decoder Model

2. DL_Assignment3_Vanilla_Wandb_Run.ipynb               ~ Wandb Hyper Parameter Tuning

3. DL_Assignment3_Vanilla_Best_Model_Train_Test.ipynb   ~ Model on Best Parameter

4. Command_Line_Run_RNN_Model.py                        ~ Command Line python code for offline runs

5. ./predictions_vanilla/predictions_vanilla.csv        ~ Contains Test Data on Vanilla Model

6. Prediction_Vanilla_Visualization.ipynb               ~ Test Prediction Sample Visualization

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

### Run Time Environment in the Command Line:

Please place the Dakshina Dataset(./dakshina_dataset_v1.0/) into the same folder where the **Command_Line_Run_RNN_Model.py** will be downloaded.

Then Please go through the below demo to run the same.

```
(base) C:\Users\joyoj\OneDrive\Desktop\DL_Final_Code>python Command_Line_Run_RNN_Model.py
2022-05-08 18:08:48.526961: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-05-08 18:08:48.527140: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2022-05-08 18:08:53.614180: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-05-08 18:08:53.617522: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'nvcuda.dll'; dlerror: nvcuda.dll not found
2022-05-08 18:08:53.617633: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2022-05-08 18:08:53.622606: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:169] retrieving CUDA diagnostic information for host: JoysDevice
2022-05-08 18:08:53.622790: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:176] hostname: JoysDevice

Target Language: Bangla, Code: bn
Target Language: Gujarati, Code: gu
Target Language: Hindi, Code: hi
Target Language: Kannada, Code: kn
Target Language: Malayalam, Code: ml
Target Language: Marathi, Code: mr
Target Language: Punjabi, Code: pa
Target Language: Sindhi, Code: sd
Target Language: Sinhala, Code: si
Target Language: Tamil, Code: ta
Target Language: Telugu, Code: te
Target Language: Urdu, Code: ur
Please Enter the Target Language Code: hi
Number of Training samples: 44204
Number of Validation samples: 4358
Number of Test samples: 4502
Number of unique input tokens: 27
Number of unique output tokens: 66
Max sequence length for inputs: 20
Max sequence length for outputs: 21
Model Building...
Hidden Layer Size (128,256,..): 64
Number of Encoder Layers (1,2,3,..): 1
Number of Decoder Layers (1,2,3,..): 1
Learning Rate (0.001,0.0001,..): 0.001
Optimizer ('adam','nadam','rmsprop'): adam
Batch Size (64,128,...): 128
Model Name ('LSTM','RNN','GRU'): LSTM
Embedding Size (64,128,...): 64
Dropouts (0.1,0.2,..): 0.3
Number of Epochs (10,20,...): 5
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, None)]       0           []

 input_2 (InputLayer)           [(None, None)]       0           []

 embedding (Embedding)          (None, None, 64)     1728        ['input_1[0][0]']

 embedding_1 (Embedding)        (None, None, 64)     4224        ['input_2[0][0]']

 lstm (LSTM)                    [(None, None, 64),   33024       ['embedding[0][0]']
                                 (None, 64),
                                 (None, 64)]

 lstm_1 (LSTM)                  [(None, None, 64),   33024       ['embedding_1[0][0]',
                                 (None, 64),                      'lstm[0][1]',
                                 (None, 64)]                      'lstm[0][2]']

 dense (Dense)                  (None, None, 66)     4290        ['lstm_1[0][0]']

==================================================================================================
Total params: 76,290
Trainable params: 76,290
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/5
346/346 [==============================] - 26s 63ms/step - loss: 1.3505 - accuracy: 0.6964 - val_loss: 1.0238 - val_accuracy: 0.7414
Epoch 2/5
346/346 [==============================] - 22s 62ms/step - loss: 0.9731 - accuracy: 0.7424 - val_loss: 0.8950 - val_accuracy: 0.7578
Epoch 3/5
346/346 [==============================] - 21s 62ms/step - loss: 0.9069 - accuracy: 0.7544 - val_loss: 0.8429 - val_accuracy: 0.7703
Epoch 4/5
346/346 [==============================] - 21s 62ms/step - loss: 0.8516 - accuracy: 0.7668 - val_loss: 0.7835 - val_accuracy: 0.7818
Epoch 5/5
346/346 [==============================] - 22s 62ms/step - loss: 0.7796 - accuracy: 0.7834 - val_loss: 0.6955 - val_accuracy: 0.8042
```


## 2. Building and Training a RNN Based Attention Model using the Google Dakshina Dataset

### Files associated:

1. DL_Assignment3_Attention_Wandb_Run.ipynb                             ~ Attention Based Model Wandb Run

2. DL_Assignment3_Attention_Best_Model_Train_Test.ipynb                 ~ Attention Based Best Model Run

3. Prediction_Attention_Visualization.ipynb                             ~ Test Samples Visualization - Attention Model

4. ./predictions_attention/predictions_attention.csv                    ~ Test Data Prediction - Attention Model

5. Attention_Vanilla_Prediction_Comparison.ipynb                        ~ Comparison Between Attention and Vanilla Model

6. nirmala.ttf                                                          ~ Font used in Attention Heatmap Plots

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

1. DL_Assignment3_Q6_Connectivity.ipynb                             ~ Python Code for Connectivity Visualization

We have visualized the different connections between input and target sequences by this part of the code.

In the Attention Model we saved the Attention Scores via the file "connectivity_visualization.txt". We used the same file to start the visualization in this part of the code.

## 4. Lyrics Generation using GPT2 given a specific string:

### Files/Folders Associated:

1. ./Question 8 GPT-2                                               ~ Python Code for GPT-2

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
