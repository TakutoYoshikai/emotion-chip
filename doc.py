import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import livedoor
import subprocess

cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')
mt=MeCab.Tagger("-d {0}".format(path))
def get_words(text):
    node = mt.parseToNode(text)
    words = []
    while node:
        surface = node.surface
        words.append(surface)
        node = node.next
    return words

def load_doc2vec_model():
    return Doc2Vec.load("model/doc2vec.model")

def train_doc2vec(texts):
    trainings = [TaggedDocument(words=get_words(text), tags=[i]) for i, text in enumerate(texts)]
    m = Doc2Vec(documents=trainings, dm=1, vector_size=300, window=15, alpha=0.025, min_alpha=0.025, min_count=1, sample=1e-6)
    m.save("model/doc2vec.model")

def train_doc2vec_livedoor():
    train_doc2vec(livedoor.load_datasets())

#train_doc2vec(["こんにちは。僕の名前は吉開拓人です。", "安倍さん、いつもありがとう。あまり自分を追い詰めないで。"])
train_doc2vec_livedoor()
