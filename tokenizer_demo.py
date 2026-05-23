import tiktoken
enc = tiktoken.get_encoding("gpt2")
text = "Hello, Transformers!"
ids  = enc.encode(text)
print("Token IDs :", ids)
print("Decoded   :", enc.decode(ids))