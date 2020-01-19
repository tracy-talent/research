'''
将分好词的句子写入文件(每行一个句子)，然后直接使用预训练好的ELMo模型(CNN-char-embed + 2-bilstm)得到最终的编码写入到
hdf5文件中，hdf5中句子id从str(0)开始，每个句子的embedding由3层embedding组成(embedding 
layer + forward lstm + backward lstm)
ELMo usage example to write biLM embeddings for an entire dataset to
a file.
'''

import os
import h5py
from bilm import dump_bilm_embeddings

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the dataset file.
dataset_file = 'dataset_file.txt'
with open(dataset_file, 'w') as fout:
    for sentence in tokenized_context + tokenized_question:
        fout.write(' '.join(sentence) + '\n')


# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'model')
vocab_file = os.path.join(datadir, 'vocab_test.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Dump the embeddings to a file. Run this once for your dataset.
embedding_file = 'elmo_embeddings.hdf5'
dump_bilm_embeddings(
    vocab_file, dataset_file, options_file, weight_file, embedding_file
)

# Load the embeddings from the file -- here the 2nd sentence.
with h5py.File(embedding_file, 'r') as fin:
    second_sentence_embeddings = fin['0'][...]
    print(second_sentence_embeddings.shape)

