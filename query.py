import nltk
from simple_transformer.lm import LM
from simple_transformer.tokeniser import Tokeniser

lm = LM('model.pt')

text = nltk.corpus.gutenberg.raw('austen-emma.txt')
tokeniser = Tokeniser()
query = tokeniser.decode(tokeniser.encode(text)[:64]).replace('<unknown>', 'unknown_token')

print(query)
result = lm.query(query, response_len=10)
print(result)
