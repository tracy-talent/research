这篇文章是我看的第一篇端到端的神经网络架构模型，sequnce输入得到sequence输出，无需对数据进行特征工程的预处理。模型架构如下图1所示，简洁明了，但是具体实现细节还是需要细抠的，具体的实现参考了github上[tf_ner](https://github.com/guillaumegenthial/tf_ner)。

<div align="center">
    <img src="https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/research/bilstm-cnns-crf.png">
</div>

<center>Figure 1: The main architetcture of chars-cnn-lstm-crf neural network</center>

论文中使用CNN for Character-level-Representation获取文本的向量表示，并要求均匀初始化为$[-\sqrt{3/dim}，\sqrt{3/dim}]$，而在[tf_ner](https://github.com/guillaumegenthial/tf_ner)的实现中还加入了Glove.840B.300d的English word embedding与CNN处理后的character embedding进行拼接作为Bi-LSTM的输入(理解Bi-LSTM模型可以参考[understand LSTM](https://blog.csdn.net/jerr__y/article/details/58598296))，其中charater embedding是要训练的，而使用的glove embedding不可作为trainable variable，这样拼接的方式能够使最终的词嵌入表示蕴含更多的信息，丰富特征。Bi-STM处理过后再输入到CRF中进行解码输出，若为训练阶段需要将真实标记输入到CRF计算log-likehood作为损失值进行优化。CNN提取词的字符表示模型图如下图2：

<div align="center">
    <img src="https://raw.githubusercontent.com/tracy-talent/Notes/master/imgs/research/chars-cnn.png">
</div>

<center>Figure 2: The convolutional neural network for extracting character-level representations of words.Dashed arrows indicate a dropout layer applied before character embeddings are input to CNN.</center>

最终，NER实验在CoNLL-2003数据集上结果

| loss      | acc       | precision | recall     | f1        |
| --------- | --------- | --------- | ---------- | --------- |
| 0.9443114 | 0.9803926 | 0.9064165 | 0.92295367 | 0.9146103 |