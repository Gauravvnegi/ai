import tiktoken

enc = tiktoken.get_encoding("o200k_base")

text = "Hello i am  Hello"
tokens = enc.encode(text)

print(tokens)          # token ids
print(len(tokens))     # token count
print(enc.decode(tokens))

# import tiktoken

# text = "Tokenization is different."

# enc1 = tiktoken.get_encoding("cl100k_base")
# enc2 = tiktoken.get_encoding("p50k_base")

# print(len(enc1.encode(text)))
# print(len(enc2.encode(text)))