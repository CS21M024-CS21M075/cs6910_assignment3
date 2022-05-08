#imports
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
print(tf.test.gpu_device_name())


#All functions
#Wordwise Inference Mechanism for Vanilla Approach
#Sigmoid Function
def sigmoid(i):
    return [1/(1 + np.exp(-z)) for z in i]

# Vanilla_Inference Function
def vanilla_inference(model, dev_encoder_input_data, dev_input, dev_target, num_decoder_tokens, max_decoder_seq_length, target_token_index, inverse_target_token_index, encoder_latent_dim, decoder_latent_dim, model_name):
    
    #Prediction Function --> Wordwise
    def decode_sequence_predict(input_sequence):
        # Encode the input as state vectors.
        states_value = [encoder_model.predict(input_sequence)] * len(decoder_models_index)
        # Generate empty target sequence of length 1.
        target_sequence = np.zeros(( 1, 1))

        # Populate the first character of target sequence with the start character.
        target_sequence[0, 0 ] = target_token_index["\t"]
        
        flag = True
        output_sequence = ""

        while flag:
            output = decoder_model.predict([target_sequence] + states_value)
            output_tokens, states_value = output[0], output[1:]

            # Sample a token
            sample_token_index = np.argmax(output_tokens[0, -1, :])
            sample_chararcter = inverse_target_token_index[sample_token_index]
            output_sequence += sample_chararcter
            if sample_chararcter == "\n" or len(output_sequence) > max_decoder_seq_length:
                flag = False
            target_sequence = np.zeros((1, 1))
            target_sequence[0, 0] = sample_token_index 
        return output_sequence
    
    
    no_of_encoder_layers = len(encoder_latent_dim)
    encoder_embedding_index, encoder_models_index = -1, []
    decoder_embedding_index, decoder_models_index = -1, []
    dense_index = -1
    encoder_layers_count = 0
    
    flag = True
    for idx, layer in enumerate(model.layers):
        print(layer.name)
        # Dense Layer
        if "dense" in layer.name :
            dense_index = idx

        # Embedding layer
        if "embedding" in layer.name:
            if flag:
                encoder_embedding_index = idx
                flag = False
            else:
                decoder_embedding_index = idx

        # Encoder-Decoder Model Layers 
        if model_name.lower() in layer.name:
            if encoder_layers_count < no_of_encoder_layers:
                encoder_models_index.append(idx)
                encoder_layers_count += 1
            else:
                decoder_models_index.append(idx)

    
    # Encoder Model
    encoder_inputs = model.input[0]  # input_1

    if model_name == "RNN" or model_name == "GRU":
        encoder_outputs, state = model.layers[encoder_models_index[-1]].output
        encoder_model = keras.Model(encoder_inputs, [state])
    
    elif model_name == "LSTM":
        encoder_outputs, state_h_enc, state_c_enc = model.layers[encoder_models_index[-1]].output
        encoder_model = keras.Model(encoder_inputs, [state_h_enc, state_c_enc])
    
    else:
        print("Wrong Choice of Model...")
        return

    #Decoder Model
    decoder_inputs = model.input[1]  # input_2
    decoder_outputs =  model.layers[decoder_embedding_index](decoder_inputs)

    decoder_states_inputs =  []
    decoder_states = []

    # Decoder Models
    for dec in range(len(decoder_latent_dim)):
        
        if model_name == "RNN" or model_name == "GRU":
            state = keras.Input(shape = (decoder_latent_dim[dec], ))
            current_states_inputs = [state]
            decoder_outputs, state = model.layers[decoder_models_index[dec]](decoder_outputs, initial_state = current_states_inputs)
            decoder_states += [state]

        elif model_name == "LSTM":
            state_h_dec, state_c_dec = keras.Input(shape = (decoder_latent_dim[dec],)),  keras.Input(shape = (decoder_latent_dim[dec],))
            current_states_inputs = [state_h_dec, state_c_dec]
            decoder_outputs, state_h_dec,state_c_dec = model.layers[decoder_models_index[dec]](decoder_outputs, initial_state = current_states_inputs)
            decoder_states += [state_h_dec, state_c_dec]
            
        else:
            print("Wrong Choice of Model...")
        
        decoder_states_inputs += current_states_inputs

    # Decoder Dense layer
    decoder_dense = model.layers[dense_index]
    decoder_outputs = decoder_dense(decoder_outputs)

    # Final decoder model
    decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    #Count of correct Predictions
    correct_count = 0
    #File to Laod the Prediction
    vanilla_prediction = open("predictions_vanilla.csv", "w", encoding='utf-8')
    vanilla_prediction.write("Input Sentence,Predicted Output Sentence,Original Target Sentence\n")
    for idx in range(len(dev_input)):
        if idx%50 == 0:
            print("Test at: ", idx)
        input_sequence = dev_encoder_input_data[idx : idx + 1]
        decoded_word = decode_sequence_predict(input_sequence)
        original_word = dev_target[idx][1:]
        vanilla_prediction.write(dev_input[idx] + "," + decoded_word[:-1] + "," + original_word[:-1] + "\n")
        if(original_word == decoded_word):
            correct_count += 1

    return correct_count/len(dev_input)


#main

print("Target Language: Bangla, Code: bn")
print("Target Language: Gujarati, Code: gu")
print("Target Language: Hindi, Code: hi")
print("Target Language: Kannada, Code: kn")
print("Target Language: Malayalam, Code: ml")
print("Target Language: Marathi, Code: mr")
print("Target Language: Punjabi, Code: pa")
print("Target Language: Sindhi, Code: sd")
print("Target Language: Sinhala, Code: si")
print("Target Language: Tamil, Code: ta")
print("Target Language: Telugu, Code: te")
print("Target Language: Urdu, Code: ur")

target_language = input("Please Enter the Target Language Code: ")
DATAPATH = "dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv"

#Defining training, validation and test path and reading the data from dataset.

#Training
train_path = DATAPATH.format(target_language, target_language, "train")
train_data = pd.read_csv(train_path, sep = '\t', header = None)

#Validation
dev_path = DATAPATH.format(target_language, target_language, "dev")
dev_data = pd.read_csv(dev_path, sep = '\t', header = None)

#Test
test_path = DATAPATH.format(target_language, target_language, "test")
test_data = pd.read_csv(test_path, sep = '\t', header = None)

#Spliting the dataset into wordwise and characterwise
#All unique characters
input_characters = set()
target_characters = set()
input_characters.add(' ')
target_characters.add(' ')

#Training Data
train_input = [str(w) for w in train_data[1]]
train_target = ["\t" + str(w) + "\n" for w in train_data[0]]
for word in train_input:
    for char in word:
        input_characters.add(char)
for word in train_target:
    for char in word:
        target_characters.add(char)

#Validation Data
dev_input = [str(w) for w in dev_data[1]]
dev_target = ["\t" + str(w) + "\n" for w in dev_data[0]]
for word in dev_input:
    for char in word:
        input_characters.add(char)
for word in dev_target:
    for char in word:
        target_characters.add(char)

#Test Data
test_input = [str(w) for w in test_data[1]]
test_target = ["\t" + str(w) + "\n" for w in test_data[0]]

for word in test_input:
    for char in word:
        input_characters.add(char) 
for word in test_target:
    for char in word:
        target_characters.add(char)
        
#Sorting the characters
input_characters = list(input_characters)
target_characters = list(target_characters)
input_characters.sort()
target_characters.sort()

#Fetching character and maximum sequence length
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max(max([len(text) for text in train_input]),max([len(text) for text in dev_input]))
max_encoder_seq_length = max(max_encoder_seq_length,max([len(text) for text in test_input]))
                             
max_decoder_seq_length = max(max([len(text) for text in train_target]),max([len(text) for text in dev_target]))
max_decoder_seq_length = max(max_decoder_seq_length,max([len(text) for text in test_target]))
                             
print("Number of Training samples:", len(train_input))
print("Number of Validation samples:", len(dev_input))
print("Number of Test samples:", len(test_input))
                             
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

#Dictionary Indexing and Inverse Dictionary Indexing for the unique Characters
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
inverse_input_token_index = dict([(i, char) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
inverse_target_token_index = dict([(i, char) for i, char in enumerate(target_characters)])

#Training Encoder-Decoder One Hot Data Preparation
train_encoder_input_data = np.zeros((len(train_input), max_encoder_seq_length), dtype="float32")
train_decoder_input_data = np.zeros((len(train_input), max_decoder_seq_length), dtype="float32")
train_decoder_target_data = np.zeros((len(train_input), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
for i, (input_text, target_text) in enumerate(zip(train_input, train_target)):
    for t, char in enumerate(input_text):
        train_encoder_input_data[i, t] = input_token_index[char]
    train_encoder_input_data[i, t + 1 :] = input_token_index[' ']
    for t, char in enumerate(target_text):
        train_decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            train_decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    train_decoder_input_data[i, t + 1 :] = target_token_index[' ']
    train_decoder_target_data[i, t:, target_token_index[' ']] =  1.0

#Validation Encoder-Decoder One Hot Data Preparation
dev_encoder_input_data = np.zeros((len(dev_input), max_encoder_seq_length), dtype="float32")
dev_decoder_input_data = np.zeros((len(dev_input), max_decoder_seq_length), dtype="float32")
dev_decoder_target_data = np.zeros((len(dev_input), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
for i, (input_text, target_text) in enumerate(zip(dev_input, dev_target)):
    for t, char in enumerate(input_text):
        dev_encoder_input_data[i, t] = input_token_index[char]
    dev_encoder_input_data[i, t + 1 :] = input_token_index[' ']
    for t, char in enumerate(target_text):
        dev_decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            dev_decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    dev_decoder_input_data[i, t + 1 :] = target_token_index[' ']
    dev_decoder_target_data[i, t:, target_token_index[' '] ] = 1.0

#Test Data setup
test_encoder_input_data = np.zeros((len(test_input), max_encoder_seq_length), dtype="float32")
for i, input_word in enumerate(test_input):
    for t, char in enumerate(input_word):
        test_encoder_input_data[i, t] = input_token_index[char]
    test_encoder_input_data[i, t + 1 :] = input_token_index[' ']


#Model Building

#Best Configuration
print("Model Building...")
hidden_layer_size=int(input("Hidden Layer Size (128,256,..): "))
num_encoder_layers=int(input("Number of Encoder Layers (1,2,3,..): "))
num_decoder_layers=int(input("Number of Decoder Layers (1,2,3,..): "))
learning_rate=float(input("Learning Rate (0.001,0.0001,..): "))
optimizer=str(input("Optimizer ('adam','nadam','rmsprop'): "))
batch_size=int(input("Batch Size (64,128,...): "))
model_name = str(input("Model Name ('LSTM','RNN','GRU'): "))
embedding_size = int(input("Embedding Size (64,128,...): "))
dropout = float(input("Dropouts (0.1,0.2,..): "))
epochs = int(input("Number of Epochs (10,20,...): "))
beam_size = 0

#Encoder Model
encoder_inputs = keras.Input(shape=(None, ))
encoder_outputs = keras.layers.Embedding(input_dim = num_encoder_tokens,
                                        output_dim = embedding_size,
                                        input_length = max_encoder_seq_length)(encoder_inputs)
    
encoder_latent_dim = [hidden_layer_size]*num_encoder_layers
for latent_dim in encoder_latent_dim:
    if model_name == "RNN":
        encoder_outputs, state = keras.layers.SimpleRNN(latent_dim, dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
        encoder_states = [state]
    elif model_name == "LSTM":
        encoder_outputs, state_h, state_c = keras.layers.LSTM(latent_dim, dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
        encoder_states = [state_h, state_c]
    elif model_name == "GRU":
        encoder_outputs, state = keras.layers.GRU(latent_dim, dropout = dropout, return_state = True, return_sequences = True)(encoder_outputs)
        encoder_states = [state]
    else:
        print("Wrong Choice of Model...")

#Decoder Model
decoder_inputs = keras.Input(shape=(None, ))
decoder_outputs = keras.layers.Embedding(input_dim = num_decoder_tokens,
                                         output_dim = embedding_size, 
                                         input_length = max_decoder_seq_length)(decoder_inputs)

decoder_latent_dim = [hidden_layer_size]*num_decoder_layers
for latent_dim in decoder_latent_dim:
    if model_name == "RNN":
        decoder = keras.layers.SimpleRNN(latent_dim, dropout = dropout, return_sequences = True, return_state = True)
        decoder_outputs, _ = decoder(decoder_outputs, initial_state = encoder_states)

    elif model_name == "LSTM":
        decoder = keras.layers.LSTM(latent_dim, dropout = dropout, return_sequences = True, return_state = True)
        decoder_outputs, _, _ = decoder(decoder_outputs, initial_state = encoder_states)

    elif model_name == "GRU":
        decoder = keras.layers.GRU(latent_dim, dropout = dropout, return_sequences = True, return_state = True)
        decoder_outputs, _= decoder(decoder_outputs, initial_state = encoder_states)
    else:
        print("Wrong Model Choice")

#Decoder Dense Layer
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Runnable Model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

#Different Optimizers
if optimizer == 'adam':
    model.compile(optimizer = Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics = ['accuracy'])
elif optimizer == 'nadam':
    model.compile(optimizer = Nadam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics = ['accuracy'])
elif optimizer == 'rmsprop':
    model.compile(optimizer = RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy", metrics = ['accuracy'])
else:
    print("Wrong Optimizer Choice...")
        
#Model fitting with train and validation data characterwise
model.fit(
    [train_encoder_input_data, train_decoder_input_data],
    train_decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_data = ([dev_encoder_input_data, dev_decoder_input_data], dev_decoder_target_data),
)
model.save("Best_Model")
validation_accuracy = vanilla_inference(model, dev_encoder_input_data, dev_input, dev_target, num_decoder_tokens, max_decoder_seq_length, target_token_index, inverse_target_token_index, encoder_latent_dim, decoder_latent_dim, model_name)
print("Wordlevel Validation Accuracy: ", validation_accuracy)

# Test Accuracy on the Best Model
test_accuracy = vanilla_inference(model, test_encoder_input_data, test_input, test_target, num_decoder_tokens, max_decoder_seq_length, target_token_index, inverse_target_token_index, encoder_latent_dim, decoder_latent_dim, model_name)
print("Wordlevel Test Accuracy: ", test_accuracy)