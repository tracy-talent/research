from corpus import Corpus
from glove import Glove

if __name__ == '__main__':
    # 读取出现频次最高的400k个词作为词汇表
    vocab = []
    with open('../../output/vocabs_100.txt', 'r') as vbf:
        for line in vbf.readlines():
            vocab.append(line.strip())

    # 建立词典，统计共现矩阵
    dictionary = {}
    for i, word in enumerate(vocab):
        dictionary[word] = i
    corpus = []
    with open('../../input/wiki.500.txt', 'r') as cf:
        for line in cf.readlines():
            corpus.append([word for word in line.split()])
    corpus_obj = Corpus(dictionary=dictionary)
    corpus_obj.fit(corpus, window=10, ignore_missing=True) # 得到稀疏的上三角矩阵
    corpus_obj.save('../../output/corpus_obj')
    # corpus_obj = Corpus.load('../../output/corpus_obj') # self.dictionary, self. matrix
    
    glove = Glove(no_components=100, learning_rate=0.05, alpha=0.75, 
    max_count=1000, max_loss=10.0, random_state=None)
    glove.fit(corpus_obj.matrix, epochs=100, no_threads=6, verbose=True)
    glove.add_dictionary(dictionary=dictionary)
    wordvectors = glove.word_vectors.round(decimals=6)
    with open('../../output/glove100.wv', 'w') as  wvf:
        for i, wv in enumerate(wordvectors):
            wvf.write(vocab[i] + ' ' + str(list(wv))[1:-1].replace(', ', ' ') + '\n')


