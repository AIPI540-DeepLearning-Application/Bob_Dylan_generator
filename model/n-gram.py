import pandas as pd
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams, FreqDist
nltk.download('punkt')
nltk.download('stopwords')

class Ngram:
    def __init__(self):
        self.df = self.load_data()
        self.tokens = self.tokensize(self.df)
        self.bi_grams = self.generate_bigrams(self.tokens)

    def load_data(self, file_path = 'Bob_Dylan.csv'):
        # load data
        df = pd.read_csv(file_path, usecols=['title', 'lyrics'])
        return df


    def tokensize(self, df):
        # combined all the lyrics into 1 string
        all_lyrics = ' '.join(df['lyrics'])

        # tokenize
        tokens = word_tokenize(all_lyrics.lower())

        # remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        return tokens
    

    def generate_bigrams(self, tokens):
        # generate bigrams
        bi_grams = list(bigrams(tokens))

        # compute the distribution of the bigram
        bi_gram_freq = FreqDist(bi_grams)

        return bi_grams
    

    def generate_text(self, bi_grams):
        # choose a random bigram as the start point
        current_bigram = random.choice(bi_grams)
        text = [current_bigram[0], current_bigram[1]]

        # generate following words
        for _ in range(50):  # generate 50 words
            next_words = [pair[1] for pair in bi_grams if pair[0] == current_bigram[1]]
            if not next_words:
                break
            next_word = random.choice(next_words)
            text.append(next_word)
            current_bigram = (current_bigram[1], next_word)

        return text

if __name__ == '__main__':
    ngram = Ngram()
    text = ngram.generate_text(ngram.bi_grams)
    print(' '.join(text))