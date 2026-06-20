
class CharacterTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, l):
        return ''.join([self.itos.get(i, ' ') for i in l])