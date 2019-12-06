# Transformer(Attention is All You Need)

> 运行环境：
>
> tensorflow 2.0.0
>
> tensorflow-datasets 1.3.0
>
> numpy 1.17.2
>
> nltk 3.4.5
>
> matplotlib 3.1.

src目录下文件说明:

* transformer.py来源于[tensorflow official tutorial: transformer](https://www.tensorflow.org/tutorials/text/transformer#decoder_layer)，而transformer.ipynb是其对应的notebook
* transformer_beam_bleu.py在transformer.py基础上加入了beam search和bleu score(调用nltk)评估
* transformer_lsr_beam_bleu.py在transformer_beam_bleu.py基础上加入了label smotthing reqularization(lsr)

以上任何一个源文件都可以单独运行，模型每5个epoches保存一次checkpoint