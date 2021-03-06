'''
from officail tensorflow guide:Transformer model for language understanding
link: https://tensorflow.google.cn/tutorials/text/transformer#setup_input_pipeline
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate import bleu_score

## Setup input pipeline
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
                              
train_examples, val_examples, test_examples = examples['train'], examples['validation'], examples['test']

# target_vocab_size:approximate size of the vocabulary to create
# 注意是大概的size，结果可能约大于size，同时target_vocab_size必须大于等于257(256个acsii码＋padding_index(0))
# 因此encode得到的token索引是从１开始的，0是padding index
# tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    # (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)
# tokenizer_en.save_to_file('vocab_file_en')
tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('vocab_file_en') # vocab_size=文件内subword数目+257
# tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    # (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
# tokenizer_pt.save_to_file('vocab_file_pt')
tokenizer_pt = tfds.features.text.SubwordTextEncoder.load_from_file('vocab_file_pt') # vocab_size=文件内subword数目+257


# model hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1
lsr_rate = 0.1 # label smooth regularization rate

# beam serach parameters
beam_size = 2 # 集束宽度beam width
candidates_num = 3 # 集束搜索的候选翻译个数，搜索到candidates_num个翻译则结束搜索，取平均长度概率最高的作为最终的结果
bleu_ngrams = 2 # 计算bleu使用几种ngrams

# dataset hyperparams
BUFFER_SIZE = 20000  # shuffle buffer size
BATCH_SIZE = 64 
MAX_LENGTH = 40 # 筛选字符最长40，To keep this example small and relatively fast, drop examples with a length of over 40 tokens.


# Add a start and end token to the input and target.
def encode(lang1, lang2):
  lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
      lang1.numpy()) + [tokenizer_pt.vocab_size+1]

  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
      lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
  return lang1, lang2

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

# transform python function:encode to tensorflow graph mode
def tf_encode(pt, en):
  return tf.py_function(encode, [pt, en], [tf.int32, tf.int32])

train_dataset = train_examples.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]), padding_values=(0, 0))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]), padding_values=(0, 0)).prefetch(
      tf.data.experimental.AUTOTUNE)

test_dataset = test_examples.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]), padding_values=(0, 0)).prefetch(
      tf.data.experimental.AUTOTUNE)


# Positional encoding
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

## test positional encode
pos_encoding = positional_encoding(50, 512)

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()



# Masking
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


# Multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # 直接将维度为d_model的embedding划分成num_heads个depth维度
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


# Point wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


# Encoder and decoder
## Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

## Decoder layer
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2

## Encoder
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)

## Decoder
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


# Create the Transformer
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    # final_output = tf.nn.softmax(self.final_layer(dec_output), axis=-1) # 如果使用了softmax要修改loss_function中的from_logits=False
    
    return final_output, attention_weights


# Optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''按论文要求自定义学习率'''
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)
## 绘制自定义学习率变化曲线
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


# Loss and metrics
# 如果是未经过sigmoid或者softmax处理过的logits则设置from_logits为True，
# 否则from_logits默认为False，这时tensorflow内部会根据概率阈值对值两极化处理
loss_object = tf.keras.losses.CategoricalCrossentropy(   
    from_logits=True, reduction='none') 

def loss_function(real, real_lsr, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real_lsr, pred)
  
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


# Training and checkpointing
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size,   # input的position encoding的最大位置，取决于最长序列有多长
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1]) # [seqlen, seqlen]
  dec_target_padding_mask = create_padding_mask(tar) # [batch_size, 1, 1, seqlen]
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask) # [batch_size, 1, seqlen, seqlen]，broadcast
  
  return enc_padding_mask, combined_mask, dec_padding_mask

checkpoint_path = "./checkpoints_lsr/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32), # 第一维batch size也要定义为None，因为dataset的最后一个batch可能不足BATCH_SIZE
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None, target_vocab_size), dtype=tf.float32)
]

# 如果不指定输入graph mode下的shape，则tf2.0后台会cache每一个输入shape不同的graph，加重内存cost
@tf.function(input_signature=train_step_signature) ## 限制输入，类似于1.版本的placeholder，注释掉也能训练
def train_step(inp, tar, real_lsr):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, real_lsr, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

EPOCHS = 20
for epoch in range(EPOCHS):
  start = time.time()
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(train_dataset):
    # label smoothing regualrization(LSR)
    real_lsr = np.full((tar.shape[0], tar.shape[1] - 1, target_vocab_size), lsr_rate * (1 / target_vocab_size), dtype=np.float32)
    # real_lsr = np.zeros((tar.shape[0], tar.shape[1] - 1, target_vocab_size), dtype=np.float32)
    # real_lsr[:] = lsr_rate * (1 / target_vocab_size)
    for i in range(tar.shape[0]):
      for j in range(tar.shape[1] - 1):
        real_lsr[i, j, tar[i][j + 1]] = 1 - lsr_rate
        
    train_step(inp, tar, real_lsr)
    
    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


# Evaluate
def beam_search(source_sentence):  
  """集束搜索
  Arguments:
    source_sentence {list} -- 有句首句尾标记的源句编码，只能是单句

  Returns:
      str -- 目标句
      tensor -- attention_weights, size:[num_heads, seq_len_q, seq_len_k]
  """

  source_sentence = tf.expand_dims(source_sentence, 0)
  seap = 1 # 当前集束搜索的路径数目search path num
  decoder_input = np.zeros((seap, 1, 1), dtype=np.int32)
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input[:] = tokenizer_en.vocab_size
  output = tf.constant(decoder_input)
  
  candidates_translation = [] # 存放beam search的翻译结果(subword索引列表)
  beam_logprob = np.zeros(beam_size, dtype=np.float32) # 当前beam_size条路径的的log probability和
  beam_next_ids = np.zeros(beam_size * beam_size, dtype=np.int32) # 从每条路径扩展beam_size个概率最大的下一词对应的索引
  beam_next_logprob = np.zeros(beam_size * beam_size, dtype=np.float32) # 从每条路径扩展beam_size个概率最大的下一词的概率值
  for i in range(MAX_LENGTH):
    beam_next_logprob[:] = -np.inf  # 进入搜索前务必初始化
    for beam_step in range(seap):
      
      enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          source_sentence, output[beam_step])
    
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = transformer(source_sentence, 
                                                  output[beam_step],
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)

      # 取下一词概率最大的beam_size个词对应的索引
      predictions = predictions.numpy()
      beam_next_ids[beam_step * beam_size : (beam_step + 1) * beam_size] = np.argpartition(predictions[: ,-1:, :], -beam_size, axis=-1)[0, -1, :-(beam_size+1):-1]
      for j in range(beam_size):
        # 更新候选搜索路径的概率值
        beam_next_logprob[beam_step * beam_size + j] = np.log(predictions[0, -1, beam_next_ids[beam_step * beam_size + j]]) + beam_logprob[beam_step]
        # 遇到终止词则将句子添加到候选翻译集合中
        if beam_next_ids[beam_step * beam_size + j] == tokenizer_en.vocab_size + 1:
          if i > 0: # 避免除0
            transidx = [idx for idx in output[beam_step][0][1:] if idx < tokenizer_en.vocab_size and idx > 0]
            if transidx != []:
              candidates_translation.append((transidx, beam_logprob[beam_step] / i))
              if (len(candidates_translation) == candidates_num): 
                break
          beam_next_logprob[beam_step * beam_size + j] = -np.inf # 结束终止词所在路径的进一步搜索
        
      if (len(candidates_translation) == candidates_num): 
        break
    
    if len(candidates_translation) == candidates_num: 
      break

    # 取beam_size * beam_size条候选路径中概率最大的beam_size条
    top_beam_ids = np.argsort(beam_next_logprob)[:-(beam_size + 1):-1] 
    seap = 0 
    for j in range(beam_size): 
      if beam_next_logprob[top_beam_ids[j]] == -np.inf: 
        break
      seap += 1
    # 根据seap个最大概率搜索路径构造下一轮预测的输入
    decoder_input = np.zeros((seap, 1, i + 2), dtype=np.int32)
    for j in range(seap):
      idx = top_beam_ids[j]
      beam_logprob[j] = beam_next_logprob[idx]
      decoder_input[j][0] = tf.concat([output[idx // beam_size][0][:], beam_next_ids[idx:idx+1]], axis=-1)
    output = tf.constant(decoder_input)

  # 处理句子长度达到MAX_LENGTH但候选翻译集合中不够候选数的情况
  for i in range(min(seap, candidates_num - len(candidates_translation))): 
    transidx = [idx for idx in output[i][0][1:] if idx < tokenizer_en.vocab_size and idx > 0]
    if transidx != []:
      candidates_translation.append((transidx, beam_logprob[i] / MAX_LENGTH))
  candidates_score = [score for sent, score in candidates_translation]
  if candidates_score != []:
    print('sentence log probability based on language model:', candidates_translation[np.argmax(candidates_score)][1])
  # return [tokenizer_en.decode([idx]) for idx in candidates_translation[np.argmax(candidates_score)][0]] if candidates_score != [] else []
  return tokenizer_en.decode([idx for idx in candidates_translation[np.argmax(candidates_score)][0]]) if candidates_score != [] else "", attention_weights


print('vocab size: en->{} pt->{}'.format(tokenizer_en.vocab_size, tokenizer_pt.vocab_size))
# bleu on test dataset
source_sent_list = []
target_sent_list = []
bleu_weights = [1 / bleu_ngrams for _ in range(bleu_ngrams)] # bleu ngrams权重
for (batch, (inp, tar)) in enumerate(test_dataset):
  starttime = time.time()
  for i in range(inp.shape[0]):
    result, _ = beam_search(inp[i])
    target = tokenizer_en.decode([idx for idx in tar[i] if idx < tokenizer_en.vocab_size and idx > 0])
    source_sent_list.append(result.split())
    target_sent_list.append([target.split()])
    print(result)
    print(target)
  print('translate one batch({} sentences) takes {} secs'.format(BATCH_SIZE, time.time() - starttime))
  # 注意：corpus_bleu不是求所有句子bleu score平均值，而是所有句子ngram统计信息累加到同一分子分母中计算
  print('bleu of {} batches on test dataset: {}'.format(batch, bleu_score.corpus_bleu(target_sent_list, source_sent_list, bleu_weights)))
print('bleu on test dataset:', bleu_score.corpus_bleu(target_sent_list, source_sent_list, bleu_weights))


def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = tokenizer_pt.encode(sentence)
  
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head + 1)
    
    # plot the attention weights
    ax.matshow(attention[head][1:, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    
    ax.set_ylim(len(result)-0.5, -0.5)
    
    ax.set_xticklabels(
        ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
                        if i < tokenizer_en.vocab_size], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()

def translate(sentence, plot=''):
  predicted_sentence, attention_weights = beam_search([tokenizer_pt.vocab_size] + tokenizer_pt.encode(sentence) + [tokenizer_pt.vocab_size + 1]) 

  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(predicted_sentence))
  
  if plot:
    plot_attention_weights(attention_weights, sentence, tokenizer_en.encode(predicted_sentence), plot)

# # test translate 1
# translate("este é um problema que temos que resolver.")
# print ("Real translation: this is a problem we have to solve .")
# ## test translate 2
# translate("os meus vizinhos ouviram sobre esta ideia.")
# print ("Real translation: and my neighboring homes heard about this idea .")
# ## test translate 3
# translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
# print ("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")
# test translate 4
translate("este é o primeiro livro que eu fiz.", plot='decoder_layer4_block2')
print ("Real translation: this is the first book i've ever done.")
