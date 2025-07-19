tok_chunk.py is what you need, an example notebook for 2048 sequence length is included if you want to copy the cells into Colab. 

Please can you test the trained tokeniser to verify that we don't have any issues :) 

```
def test_examples(tokenizer):
  def test(text):
    return ' '.join(tokenizer.encode(text).tokens)

  texts = [
   "The Northern Lights season is here... "
  ]
```
