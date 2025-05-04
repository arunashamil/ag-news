from tokenizer import tk
import re

def preprocessing(sent, print_steps=False):
    
    sent = sent.lower()
    sent = re.sub(r'[^\w\s]', '', sent)
    sent = tk(sent)
    sent = ' '.join(sent)

    return sent