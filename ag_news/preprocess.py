from collections import Counter
import string

# Text preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Build vocabulary
def build_vocab(texts, max_size=20000):
    word_counts = Counter()
    for text in texts:
        word_counts.update(preprocess_text(text))
    vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common(max_size))}
    vocab['<unk>'] = 0
    return vocab