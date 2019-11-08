# 环境要求
1. numpy
2. tensorflow
3. tqdm
4. cython
5. scipy

# 接口说明
## corpus
vectorized目录下的process_wiki_WikiCorpus.py用于提取wiki预料集中的文本内容，wikisegs.py用于对提取出的内容进行分词，dictionary.py统计分词后语料集的词频，取词频最靠前的n个作为词典用于后续词向量学习，而corpus_cython.pyx和corpus.py则根据分好词的语料库建立稀疏的上三角共现矩阵。

corpus_cython.pyx中的def construct_cooccurrence_matrix(corpus, dictionary, int supplied,
                                  int window_size, int ignore_missing):
1. window_size: 8
2. dictionary_size: 8
3. ignore_missing: True
4. corpus: 列表，列表元素为分好词的段落或文章
5. supplied: 0表示未提供字典，1表示提供字典

return: 小的wordid指向大的wordid的上三角稀疏矩阵scipy.sparse.coo_matrix，coo_matrix.row、coo_matrix.col、coo_matrix.data分别为存储行索引、列索引、数据的numpy array


## vectorized
vectorized是将所有计算向量化的方法，这使得一次不能训练太多的词汇，16G内存训练20k词汇差不多。基础类接口定义包含在mittens_base.py中，而具体的训练及梯度计算包含在np_mittens.py(numpy实现)和tf_mittens.py(tensorflow实现，会保存summary供tensorboard查看)中。
### GloVe
构造函数：

GloVe(n=100, xmax=100, alpha=0.75,
                 max_iter=100, learning_rate=0.05, tol=1e-4,
                 display_progress=10, log_dir='../output/tf_glove', log_subdir=None,
                 test_mode=False, **kwargs):

```
Parameters:
n : int (default: 100)
    The embedding dimension.
    
xmax : int (default: 100)
    Word pairs with frequency greater than this are given weight 1.0.

    Word pairs with frequency under this are given weight
    (c / xmax) ** alpha, where c is the co-occurence count
    (see the paper, eq. (9)).

alpha : float (default: 0.75)
    Exponent in the weighting function (see [1]_, eq. (9)).

learning_rate : float (default: 0.01)
    Learning rate used for the Adagrad optimizer.

tol : float (default: 1e-4)
    Stopping criterion for the loss.

max_iter : int (default: 100)
    Number of training epochs. Default: 100, as in [1]_.

log_dir : None or str (default: None)
    If `None`, no logs are kept.
    If `str`, this should be a directory in which to store
    Tensorboard logs. Logs are in fact stored in subdirectories
    of this directory to easily keep track of multiple runs.

log_subdir : None or str (default: None)
    Use this to keep track of experiments. If `None`, 1 + the
    number of existing subdirectories is used to avoid overwrites.

    If `log_dir` is None, this value is ignored.

display_progress : int (default: 10)
    Frequency with which to update the progress bar.
    If 0, no progress bar is shown.

test_mode : bool
    If True, initial parameters are stored as `W_start`, `C_start`,
    `bw_start` and `bc_start`.
```
### Mittens
构造函数：

mittens = Mittens(n=100, mittens=0.1, xmax=100, alpha=0.75,
                 max_iter=100, learning_rate=0.05, tol=1e-4,
                 display_progress=10, log_dir='../output/tf_mittens', log_subdir=None,
                 test_mode=False, **kwargs):

wordvecctors = mittens.fit(X, vocabs) 

> tips<br>
> corpus.py返回的为稀疏矩阵，且只有左上角部分<br>
> 修改:<br>
> 1. coupus_cython的计数增量与中心词距离无关，统一为1
> 2. mittens_base.py的coincidence相关的改为稀疏矩阵的形式计算(_initialize和log_of_array_ignoring_zeros方法)
> 3. np_mittens.py的coincidence相关的改为稀疏矩阵的形式计算(_get_gradients_and_error方法)

## sparsed
该方法将共现矩阵以稀疏矩阵的形式存储，节省内存，并且训练不是向量形式的总体计算，而是根据稀疏共现矩阵的内容一次计算一个上下文词对之间的损失以及梯度，这极大地节省了内存的开销，避免了向量化方法中开辟许多大矩阵的内存开销，使得同等内存下可以训练更多的词向量，训练400k词汇花费了将近4个小时。
### glove
glove.py中包含了Glove累，创建实例后调用glove的训练方法fit

在训练之后方能调用调用对象的add_dictionary加入字典

调用most_similar(word, number)方法查询与word最相似的number个词(包含该词本身)

调用trandform_paragraph(paragraph,epoches,ignore_missing)方法训练得到段落的词向量，paragraph为分好词的列表，建立paragraph初始向量为段落中所有词的平均，建立paragraph与段落中词的共现矩阵，然后采用与glove学习词向量一样的方法训练epoches轮得到最终的paragraph vector

调用most_similar_paragraph(paragraph, number)方法查询与paragraph向量最接近的number个词，paragraph为分好的词列表

调用save和load可以分别存储对象的self.__dict__和加载对象的self.\_\_dict\_\_，存储为pickle文件

如果已经有训练好的词向量，比如stanford官方提供的词向量[Glove](http://www-nlp.stanford.edu/projects/glove/)，自己训练好的词向量也可以，只要和stanford提供的词向量格式一致即可。直接调用Glove的类方法load_stanford(filename)返回包含加载进的词向量的Glove对象，可以无需训练直接调用相似度查询等方法。

### metrics
metrics目录中是词向量之间analogy(即词类比任务)的度量方式