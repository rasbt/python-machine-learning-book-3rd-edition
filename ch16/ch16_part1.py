# coding: utf-8


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os
import gzip
import shutil
from collections import Counter
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

# *Python Machine Learning 3rd Edition* by [Sebastian Raschka](https://sebastianraschka.com) & [Vahid Mirjalili](http://vahidmirjalili.com), Packt Publishing Ltd. 2019
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-3rd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt)

# # Chapter 16: Modeling Sequential Data Using Recurrent Neural Networks (Part 1/2)

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).









# # Introducing sequential data
# 
# ## Modeling sequential data⁠—order matters
# 
# ## Representing sequences
# 
# 





# ## The different categories of sequence modeling





# # RNNs for modeling sequences
# 
# ## Understanding the RNN looping mechanism
# 









# ## Computing activations in an RNN
# 









# ## Hidden-recurrence vs. output-recurrence







tf.random.set_seed(1)

rnn_layer = tf.keras.layers.SimpleRNN(
    units=2, use_bias=True, 
    return_sequences=True)
rnn_layer.build(input_shape=(None, None, 5))

w_xh, w_oo, b_h = rnn_layer.weights

print('W_xh shape:', w_xh.shape)
print('W_oo shape:', w_oo.shape)
print('b_h shape:', b_h.shape)




x_seq = tf.convert_to_tensor(
    [[1.0]*5, [2.0]*5, [3.0]*5],
    dtype=tf.float32)


## output of SimepleRNN:
output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))

## manually computing the output:
out_man = []
for t in range(len(x_seq)):
    xt = tf.reshape(x_seq[t], (1, 5))
    print('Time step {} =>'.format(t))
    print('   Input           :', xt.numpy())
    
    ht = tf.matmul(xt, w_xh) + b_h    
    print('   Hidden          :', ht.numpy())
    
    if t>0:
        prev_o = out_man[t-1]
    else:
        prev_o = tf.zeros(shape=(ht.shape))
        
    ot = ht + tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)
    out_man.append(ot)
    print('   Output (manual) :', ot.numpy())
    print('   SimpleRNN output:'.format(t), output[0][t].numpy())
    print()


# ## The challenges of learning long-range interactions
# 





# 
# ## Long Short-Term Memory cells 





# # Implementing RNNs for sequence modeling in TensorFlow
# 
# ## Project one: predicting the sentiment of IMDb movie reviews
# 
# ### Preparing the movie review data
# 
# 









with gzip.open('../ch08/movie_data.csv.gz', 'rb') as f_in, open('movie_data.csv', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)




df = pd.read_csv('movie_data.csv', encoding='utf-8')

df.tail()




# Step 1: Create a dataset

target = df.pop('sentiment')

ds_raw = tf.data.Dataset.from_tensor_slices(
    (df.values, target.values))

## inspection:
for ex in ds_raw.take(3):
    tf.print(ex[0].numpy()[0][:50], ex[1])


#  * **Train/validaiton/test splits**



tf.random.set_seed(1)

ds_raw = ds_raw.shuffle(
    50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)


#  * **Tokenizer and Encoder**
#    * `tfds.features.text.Tokenizer`: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/Tokenizer
#    * `tfds.features.text.TokenTextEncoder`: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/TokenTextEncoder

#  * **Encoding sequences: keeping the last 100 items in each sequence**



## Step 2: find unique words


try:
    tokenizer = tfds.features.text.Tokenizer()
except AttributeError:
    tokenizer = tfds.deprecated.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)
    
print('Vocab-size:', len(token_counts))




## Step 3: endoding each unique token into integers

try:
    encoder = tfds.features.text.TokenTextEncoder(token_counts)
except AttributeError:
    encoder = tfds.deprecated.text.TokenTextEncoder(token_counts)

example_str = 'This is an example!'
encoder.encode(example_str)




## Step 3-A: define the function for transformation

def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label

## Step 3-B: wrap the encode function to a TF Op.
def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], 
                          Tout=(tf.int64, tf.int64))




ds_train = ds_raw_train.map(encode_map_fn)
ds_valid = ds_raw_valid.map(encode_map_fn)
ds_test = ds_raw_test.map(encode_map_fn)

tf.random.set_seed(1)
for example in ds_train.shuffle(1000).take(5):
    print('Sequence length:', example[0].shape)
    
example


#  * **batch() vs. padded_batch()**

# ```python
# 
# # this will result in error
# 
# 
# BATCH_SIZE = 32
# train_data = all_encoded_data.batch(BATCH_SIZE)
# 
# next(iter(train_data))
# 
# # Running this will result in error
# # We cannot apply .batch() to this dataset
# ```



## Take a small subset

ds_subset = ds_train.take(8)
for example in ds_subset:
    print('Individual Shape:', example[0].shape)

## batching the datasets
ds_batched = ds_subset.padded_batch(
    4, padded_shapes=([-1], []))

for batch in ds_batched:
    print('Batch Shape:', batch[0].shape)




## batching the datasets
train_data = ds_train.padded_batch(
    32, padded_shapes=([-1],[]))

valid_data = ds_valid.padded_batch(
    32, padded_shapes=([-1],[]))

test_data = ds_test.padded_batch(
    32, padded_shapes=([-1],[]))


# ### Embedding layers for sentence encoding
# 
# 
#  * `input_dim`: number of words, i.e. maximum integer index + 1.
#  * `output_dim`: 
#  * `input_length`: the length of (padded) sequence
#     * for example, `'This is an example' -> [0, 0, 0, 0, 0, 0, 3, 1, 8, 9]`   
#     => input_lenght is 10
#  
#  
# 
#  * When calling the layer, takes integr values as input,   
#  the embedding layer convert each interger into float vector of size `[output_dim]`
#    * If input shape is `[BATCH_SIZE]`, output shape will be `[BATCH_SIZE, output_dim]`
#    * If input shape is `[BATCH_SIZE, 10]`, output shape will be `[BATCH_SIZE, 10, output_dim]`









model = tf.keras.Sequential()

model.add(Embedding(input_dim=100,
                    output_dim=6,
                    input_length=20,
                    name='embed-layer'))

model.summary()


# ### Building an RNN model
# 
# * **Keras RNN layers:**
#   * `tf.keras.layers.SimpleRNN(units, return_sequences=False)`
#   * `tf.keras.layers.LSTM(..)`
#   * `tf.keras.layers.GRU(..)`
#   * `tf.keras.layers.Bidirectional()`
#  
# * **Determine `return_sequenes=?`**
#   * In a multi-layer RNN, all RNN layers except the last one should have `return_sequenes=True`
#   * For the last RNN layer, decide based on the type of problem: 
#      * many-to-many: -> `return_sequences=True`
#      * many-to-one : -> `return_sequenes=False`
#      * ..
#     



## An example of building a RNN model
## with SimpleRNN layer


model = Sequential()
model.add(Embedding(1000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1))
model.summary()




## An example of building a RNN model
## with LSTM layer




model = Sequential()
model.add(Embedding(10000, 32))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.summary()




## An example of building a RNN model
## with GRU layer

model = Sequential()
model.add(Embedding(10000, 32))
model.add(GRU(32, return_sequences=True))
model.add(GRU(32))
model.add(Dense(1))
model.summary()


# ### Building an RNN model for the sentiment analysis task



embedding_dim = 20
vocab_size = len(token_counts) + 2

tf.random.set_seed(1)

## build the model
bi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name='embed-layer'),
    
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, name='lstm-layer'),
        name='bidir-lstm'), 

    tf.keras.layers.Dense(64, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

bi_lstm_model.summary()

## compile and train:
bi_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy'])

history = bi_lstm_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=10)

## evaluate on the test data
test_results= bi_lstm_model.evaluate(test_data)
print('Test Acc.: {:.2f}%'.format(test_results[1]*100))




if not os.path.exists('models'):
    os.mkdir('models')


bi_lstm_model.save('models/Bidir-LSTM-full-length-seq.h5')


#  * **Trying SimpleRNN with short sequences**



def preprocess_datasets(
    ds_raw_train, 
    ds_raw_valid, 
    ds_raw_test,
    max_seq_length=None,
    batch_size=32):
    
    ## Step 1: (already done => creating a dataset)
    ## Step 2: find unique tokens
    tokenizer = tfds.features.text.Tokenizer()
    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        if max_seq_length is not None:
            tokens = tokens[-max_seq_length:]
        token_counts.update(tokens)

    print('Vocab-size:', len(token_counts))


    ## Step 3: encoding the texts
    encoder = tfds.features.text.TokenTextEncoder(token_counts)
    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        if max_seq_length is not None:
            encoded_text = encoded_text[-max_seq_length:]
        return encoded_text, label

    def encode_map_fn(text, label):
        return tf.py_function(encode, inp=[text, label], 
                              Tout=(tf.int64, tf.int64))

    ds_train = ds_raw_train.map(encode_map_fn)
    ds_valid = ds_raw_valid.map(encode_map_fn)
    ds_test = ds_raw_test.map(encode_map_fn)

    ## Step 4: batching the datasets
    train_data = ds_train.padded_batch(
        batch_size, padded_shapes=([-1],[]))

    valid_data = ds_valid.padded_batch(
        batch_size, padded_shapes=([-1],[]))

    test_data = ds_test.padded_batch(
        batch_size, padded_shapes=([-1],[]))

    return (train_data, valid_data, 
            test_data, len(token_counts))




def build_rnn_model(embedding_dim, vocab_size,
                    recurrent_type='SimpleRNN',
                    n_recurrent_units=64,
                    n_recurrent_layers=1,
                    bidirectional=True):

    tf.random.set_seed(1)

    # build the model
    model = tf.keras.Sequential()
    
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name='embed-layer')
    )
    
    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers-1)
            
        if recurrent_type == 'SimpleRNN':
            recurrent_layer = SimpleRNN(
                units=n_recurrent_units, 
                return_sequences=return_sequences,
                name='simprnn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = LSTM(
                units=n_recurrent_units, 
                return_sequences=return_sequences,
                name='lstm-layer-{}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = GRU(
                units=n_recurrent_units, 
                return_sequences=return_sequences,
                name='gru-layer-{}'.format(i))
        
        if bidirectional:
            recurrent_layer = Bidirectional(
                recurrent_layer, name='bidir-'+recurrent_layer.name)
            
        model.add(recurrent_layer)

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return model






batch_size = 32
embedding_dim = 20
max_seq_length = 100

train_data, valid_data, test_data, n = preprocess_datasets(
    ds_raw_train, ds_raw_valid, ds_raw_test, 
    max_seq_length=max_seq_length, 
    batch_size=batch_size
)


vocab_size = n + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='SimpleRNN', 
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=True)

rnn_model.summary()




rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])


history = rnn_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=10)




results = rnn_model.evaluate(test_data)




print('Test Acc.: {:.2f}%'.format(results[1]*100))


# ## Optional exercise: 
# 
# ### Uni-directional SimpleRNN with full-length sequences



batch_size = 32
embedding_dim = 20
max_seq_length = None

train_data, valid_data, test_data, n = preprocess_datasets(
    ds_raw_train, ds_raw_valid, ds_raw_test, 
    max_seq_length=max_seq_length, 
    batch_size=batch_size
)


vocab_size = n + 2

rnn_model = build_rnn_model(
    embedding_dim, vocab_size,
    recurrent_type='SimpleRNN', 
    n_recurrent_units=64,
    n_recurrent_layers=1,
    bidirectional=False)

rnn_model.summary()




rnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

history = rnn_model.fit(
    train_data, 
    validation_data=valid_data, 
    epochs=10)


# # Appendix
# 

# ### A -- An alternative way to get the dataset: using tensorflow_datasets



imdb_bldr = tfds.builder('imdb_reviews')
print(imdb_bldr.info)

imdb_bldr.download_and_prepare()

datasets = imdb_bldr.as_dataset(shuffle_files=False)

datasets.keys()




imdb_train = datasets['train']
imdb_train = datasets['test']


# ### B -- Tokenizer and Encoder
# 
#  * `tfds.features.text.Tokenizer`: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/Tokenizer
#  * `tfds.features.text.TokenTextEncoder`: https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/TokenTextEncoder
# 
# 



vocab_set = {'a', 'b', 'c', 'd'}
encoder = tfds.features.text.TokenTextEncoder(vocab_set)
print(encoder)

print(encoder.encode(b'a b c d, , : .'))

print(encoder.encode(b'a b c d e f g h i z'))


# ### C -- Text Pre-processing with Keras 



TOP_K = 200
MAX_LEN = 10

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K)

tokenizer.fit_on_texts(['this is an example', 'je suis en forme '])
sequences = tokenizer.texts_to_sequences(['this is an example', 'je suis en forme '])
print(sequences)

tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN)




TOP_K = 20000
MAX_LEN = 500

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K)

tokenizer.fit_on_texts(
    [example['text'].numpy().decode('utf-8') 
     for example in imdb_train])

x_train = tokenizer.texts_to_sequences(
    [example['text'].numpy().decode('utf-8')
     for example in imdb_train])

print(len(x_train))


x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=MAX_LEN)

print(x_train_padded.shape)


# ### D -- Embedding
# 
# 





tf.random.set_seed(1)
embed = Embedding(input_dim=100, output_dim=4)

inp_arr = np.array([1, 98, 5, 6, 67, 45])
tf.print(embed(inp_arr))
tf.print(embed(inp_arr).shape)

tf.print(embed(np.array([1])))


# 
# ---

# 
# 
# Readers may ignore the next cell.
# 




