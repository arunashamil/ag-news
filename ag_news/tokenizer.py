from torchtext.data.utils import get_tokenizer

tk = get_tokenizer('basic_english')

def get_tokenized_sentences(sentences):
    for sent in sentences:
        yield tk(sent)

def convert_sent_to_nums(sent, vocab, print_steps=False):
    if print_steps is True:
        print(f'Sentence (as it is):\n{sent}\n')
    return vocab(tk(sent))