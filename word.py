from gensim.models import Word2Vec
import MeCab
import livedoor

def train(sentences):
    num_features = 200
    min_word_count = 20
    num_workers = 40
    context = 10
    downsampling = 1e-3
    return Word2Vec(sentences, workers=num_workers, hs=0, sg=1, negative=10, iter=25, size=num_features, min_count=min_word_count, window=context, sample=downsampling, seed=1)

model_name = "word2vec.model"
def save_model(model):
    model.save("./model/" + model_name)

def load_model():
    return Word2Vec.load("./model/" + model_name)

def train_livedoor():
    sentences = livedoor.load_datasets()
    model = train(sentences)
    save_model(model)
    return model

train_livedoor()
