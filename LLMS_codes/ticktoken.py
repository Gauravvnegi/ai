# import tiktoken

# enc = tiktoken.get_encoding("o200k_base")

# text = "Hello i am  Hello"
# tokens = enc.encode(text)
import tiktoken
enc = tiktoken.get_encoding("p50k_base")
text= "Gaurav"
tokens = enc.encode(text)
print(tokens)          # token ids

# print(tokens)          # token ids
# print(len(tokens))     # token count
# print(enc.decode(tokens))

# import tiktoken

# text = "Tokenization is different."

# enc1 = tiktoken.get_encoding("cl100k_base")
# enc2 = tiktoken.get_encoding("p50k_base")

# print(len(enc1.encode(text)))
# print(len(enc2.encode(text)))

import tiktoken

# GPT-4o-mini pricing (approx — you can adjust if OpenAI updates it)
INPUT_COST_PER_1M = 0.15   # $0.15 per 1M tokens
OUTPUT_COST_PER_1M = 0.60  # assumed same text generation rate (for estimation)

def analyze_text(text: str, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)

    tokens = enc.encode(text)
    token_count = len(tokens)

    # 💰 cost estimate (input-only approximation)
    cost = (token_count / 1000000) * INPUT_COST_PER_1M

    # 📏 context checks
    context_limits = {
        "8k": 8000,
        "32k": 32000,
        "128k": 128000
    }

    fits = {
        key: token_count <= limit
        for key, limit in context_limits.items()
    }

    return {
        "tokens": token_count,
        "estimated_cost_usd": round(cost, 8),
        "fits_context": fits
    }


# ----------------------------
# 🧪 TEST CASES
# ----------------------------

tweet = "AI is changing the world 🚀"

paragraph = """
Artificial Intelligence is transforming industries by enabling machines to learn from data,
make predictions, and automate complex tasks. It is widely used in healthcare, finance,
education, and many other domains.
"""

wikipedia_style_text = """
Machine learning is a subset of artificial intelligence that focuses on building systems
that can learn from data. The field of machine learning has evolved significantly since
its inception in the mid-20th century. Today, it powers recommendation systems, search
engines, autonomous vehicles, and natural language processing applications. Researchers
continue to develop more efficient algorithms to improve accuracy and scalability.
""" * 20   # simulate long article


# Run tests
for name, txt in [("Tweet", tweet), ("Paragraph", paragraph), ("Wikipedia Article", wikipedia_style_text)]:
    result = analyze_text(txt)

    print("\n======================")
    print(name)
    print("======================")
    print("Tokens:", result["tokens"])
    print("Cost ($):", result["estimated_cost_usd"])
    print("Fits 8k:", result["fits_context"]["8k"])
    print("Fits 32k:", result["fits_context"]["32k"])
    print("Fits 128k:", result["fits_context"]["128k"])