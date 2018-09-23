import gensim
from docx import Document
import os
from gensim.models import Word2Vec
import logging
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk.data
# nltk.download('punkt')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')


def read_docs(file):
    data = Document(file)
    raw_text = ''
    for p in data.paragraphs:
        raw_text += p.text + ' '
    return raw_text
    # f = open("path/to/text/file", "r+", encoding = 'utf8')
    # q = f.read()


def pre_processing(case):
    text = []
    path = os.path.join(os.getcwd(), 'data', case)
    stop_words = set(stopwords.words('english')) # language can change
    for idx, file in enumerate(os.listdir(path)):
        if file.endswith('docx'):
            word_tok = word_tokenize(read_docs(os.path.join(path,file))) # splits the text into a list of words
            remove_stopwords = [w for w in word_tok if not w in stop_words]
            sent = ' '.join(remove_stopwords) # joins the lists of words back into a string
            text.append(sent)
    return text

# res = pre_processing('afc')
# for doc in res:
#     print(doc)
#     break
# print(len(sent)) #Confirms that you have filtered words
# print(len(q))


def sentence_to_wordlist(sentences, remove_stopwords=False):
    # 1. Remove non-letters
    sentence_text = re.sub(r'[^\w\s]', '', sentences)  # removes all non-alphanumerics characters
    # 2. Convert words to lower case and split them
    words = sentence_text.lower().split()
    # 3. Return a list of words
    return words


def split_by_sentences(tokenizer, remove_stopwords=False):
    for doc in pre_processing('fr'):
        try:
            # 1. Use the NLTK tokenizer to split the text into sentences
            raw_sentences = tokenizer.tokenize(doc.strip())
            # 2. Loop over each sentence
            sentences = []

            for raw_sentence in raw_sentences:
                # If a sentence is empty, skip it
                if len(raw_sentence) > 0:
                    # Otherwise, call sentence_to_wordlist to get a list of words
                    sentences.append(sentence_to_wordlist(raw_sentence))
            # 3. Return the list of sentences (each sentence is a list of words, so this returns a list of lists)
            len(sentences)
            return sentences
        except:
            print('nope')


sent_list = split_by_sentences(tokenizer)

# # These are parameters that can be tweaked to accomodate the corpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 100    # Word vector dimensionalty
min_word_count = 1   # Minimum word count (ie. Only count words that show at least the min times)
num_workers = 4       # Number of threads to run in parallel
context = 20         # Context window size (ie. how many words surrounding the target word will count towards)
downsampling = 1e-3

model = gensim.models.Word2Vec(sent_list, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

model.save('fr_model_w20') #save the model for future use or analysis
# #load model
# # model = Word2Vec.load('model name')

# print(model) #to see model vocab size
# print(model.wv.vocab) #display model vocab

# model.most_similar(positive='', topn=30) #view the words that have the highest correlation to the target word
# model.most_similar(positive='', negative='') #view words that are related to the (positive) but not related to the (negative) 

#For (positive) and (negative) string args can be lists as well
# Ex: model.most_similar(positive=['',''], topn=30)
# Ex: model.most_similar(positive=['',''], negative=['','',''])

### Other examples of post model analysis functions
### https://radimrehurek.com/gensim/models/word2vec.html
