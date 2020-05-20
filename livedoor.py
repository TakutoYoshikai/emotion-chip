
import glob
import os

dataset_dir = "./datasets/livedoor/"
dataset_files = list(map(lambda s:dataset_dir + s + "/*.txt", ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme", "movie-enter","peachy","smax","sports-watch","topic-news"]))


def load_dataset(path):
    files = glob.glob(path)
    print(files)
    sentences = []
    for file_path in files:
        f = open(file_path)
        data = f.read()
        f.close()
        _sentences = data.split("\n")[3:]
        sentences += _sentences
    return sentences

def load_datasets():
    result = []
    for path in dataset_files:
        result += load_dataset(path)
    return result
