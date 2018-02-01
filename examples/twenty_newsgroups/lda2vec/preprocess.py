import logging
import pickle
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
import re

import numpy as np
from lda2vec import preprocess
from lda2vec.corpus import Corpus
from nltk.corpus import stopwords

import config as cfg

# from lda2vec import Corpus

logging.basicConfig()

# Fetch data
remove = ('headers', 'footers', 'quotes')


def read_data(path):
    with open(path) as f:
        content = f.readlines()
    content = [(x.split('\tX\t')[1].strip()) for x in content]
    return content

# texts = fetch_20newsgroups(subset='train', remove=remove).data
texts = read_data(cfg.data_file_to_process)[:1]
# Remove tokens with these substrings
stopwords = set(stopwords.words('english'))
bad = set(['--','–','>', '<', '{', '}', '[', ']', '(', ')', '^', '+', '—a', '"', ':', ';', ' –'])
#
print('got text')
def clean(line):
    # line = re.sub(' +',' ',line)
    line = ' '.join(w for w in line.split() if not any(t == w.strip() for t in stopwords))
    line = ' '.join(w for w in line.split() if not any(t in w.strip() for t in bad))
    line = ' '.join([x for x in line.split() if not any(c.isdigit() for c in x.strip())])
    return line

# Preprocess data
max_length = 1000   # Limit of 10k words per document
# Convert to unicode (spaCy only works with unicode)
print('cleaning text')
texts = [(clean(d)) for d in texts]
print('finished cleaning text')

print('start tokenizer')
tokens, vocab = preprocess.tokenize(texts, max_length, merge=False, n_threads=4, parse=False, entity=False)
print('finished tokenizer')
corpus = Corpus()
# Make a ranked list of rare vs frequent words
corpus.update_word_count(tokens)
corpus.finalize()
# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This builds a new compact index
compact = corpus.to_compact(tokens)
# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=20)
# Convert the compactified arrays into bag of words arrays
bow = corpus.compact_to_bow(pruned)
# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)
# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
assert flattened.min() >= 0
# Fill in the pretrained word vectors
n_dim = 300
print('calculating docs vectors using w2v model')
fn_wordvc = cfg.w2v_model_path
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)
print('finished calculating vectors')
# Save all of the preprocessed files

data_dir = cfg.data_dir
print('generating all pkl files')
pickle.dump(vocab, open(data_dir+'/vocab.pkl', 'wb'))
pickle.dump(corpus, open(data_dir+'/corpus.pkl', 'wb'))
np.save(data_dir+"/flattened", flattened)
np.save(data_dir+"/doc_ids", doc_ids)
np.save(data_dir+"/pruned", pruned)
np.save(data_dir+"/bow", bow)
np.save(data_dir+"/vectors", vectors)
print('finished generating files')