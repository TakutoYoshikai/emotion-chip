import MeCab
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def get_words(text):
    mt = MeCab.Tagger("mecabrc")
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

train_doc2vec(["こんにちは。僕の名前は吉開拓人です。", "安倍さん、いつもありがとう。あまり自分を追い詰めないで。"])
