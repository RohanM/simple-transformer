import nltk
from simple_transformer.lm import LM

lm = LM('model.pt')

result = lm.query(' ', response_len=256)
print(result)
