import os
from dotenv import load_dotenv
from google import genai
from google.genai import types 

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain how AI works in a few words",
     config=types.GenerateContentConfig(  # ← temperature goes here
        temperature=2,
    )
)

print(response.text)