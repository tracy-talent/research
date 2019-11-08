#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author:  liujian
@Email:  brooksj@foxmail.com
@File:  Word2Vec.py
@Version:  1.0
@DateTime:  2019/08/02 10:44:43
'''

# here put the import lib

import os
import sys
# warning: 使用完整路径,否则config中配置的文件路径不完整
sys.path.append(os.path.abspath('../'))
from config import INPUT_PATH

import collections
import math

import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import config

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size.
  
  Arguments:
      filename {str} -- 下载文件到本地的完整路径
      expected_bytes {int} -- 文件应包含的字节数用于校验
  
  Raises:
      Exception: 下载的文件校验失败
  
  Returns:
      str -- 返回下载的文件完整路径名
  """
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename[filename.rfind('/') + 1:], filename)
  statinfo = os.stat(filename)  # get file status
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download(os.path.join(INPUT_PATH, 'text8.zip'), 31344016)


# Read the data into a list of strings.
def read_data(filename):
  """从zip压缩文件中读取数据
  
  Arguments:
      filename {str} -- 要读取的压缩文件的路径
  
  Returns:
      list -- 以词为单个元素的词汇列表
  """
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

words = read_data(filename)
print('Data size', len(words))


# 词汇表大小,取频数高的
vocabulary_size = 50000

def build_dataset(words):
  """从原始文本语料库中建立词汇表,词索引,词编号,词频统计
  
  Arguments:
      words {list} -- 原始语料库的词列表
  
  Returns:
      data {list} -- 语料库中的词编号列表
      count {list(tuple)} -- tuple(word,freq)的列表
      dictionary {dict} -- 词汇表:存储词索引的字典
      reverse_dictionary {dict} -- dictionary的反转形式
  """
  # 词汇表中词的频数统计
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  # 词汇表,为词建立索引
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  # 根据词汇表索引映射语料库中每个词的编号
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

# 对原始语料库进行预处理的统计
data, count, dictionary, reverse_dictionary = build_dataset(words)

# 删除原始单词列表以节省内存
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]]) 


# 全局的单词序号,方便从训练数据中循环生成batch
data_index = 0

# 生成batch
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1
  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j][0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels


# 超参数
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
  train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
  train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
  
  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                      biases=nce_biases,
                                      labels=train_labels,
                                      inputs=embed,
                                      num_sampled=num_sampled,
                                      num_classes=vocabulary_size))
  
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keepdims=True))
  normalized_embeddings = (embeddings - tf.reduce_mean(embeddings, 1, keepdims=True)) / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
  init.run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      print('Average loss at step', step, ':', average_loss)
      average_loss = 0
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8
        nearest = (-sim[i, :]).argsort()[1:top_k+1]  # 除去自己最相似的前8个
        log_str = "Nearest to %s:" % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.show()
  plt.savefig(filename)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
poly_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:poly_only,:])
labels = [reverse_dictionary[i] for i in range(poly_only)]
plot_with_labels(low_dim_embs, labels)
