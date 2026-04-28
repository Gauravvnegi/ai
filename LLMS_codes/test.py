from groq import Groq
from dotenv import load_dotenv
import os

# load env file
load_dotenv()

# get real key from env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "Give me an example of a loop in Python."},
    ]
)

print("Answer:")
print(response.choices[0].message.content)

print("\nFull Response:")
print(response)