from transformers import pipeline

# 1. Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
reviews = [
    "Amazing movie!", "Worst film ever", "Loved the acting",
    "Too boring", "Fantastic story", "Waste of time",
    "Great direction", "Not good", "Superb visuals", "Bad ending"
]
print("Sentiment Analysis:")
for r in reviews:
    print(r, "->", sentiment(r)[0])

# 2. Summarization
# summarizer = pipeline("summarization")
# article = """Artificial Intelligence is transforming industries by automating tasks,
# improving efficiency, and enabling new innovations. Many companies are adopting AI
# to gain competitive advantages. However, challenges such as bias, data privacy,
# and ethical concerns remain important issues to address."""
# print("\nSummary:")
# print(summarizer(article, max_length=50, min_length=20)[0]['summary_text'])

# 3. Translation (English → French)
translator = pipeline("translation_en_to_fr")
print("\nTranslation:")
print(translator("AI is changing the world")[0]['translation_text'])

# 4. Named Entity Recognition (NER)
ner = pipeline("ner")
news = "Elon Musk visited India and met officials in New Delhi."
print("\nNamed Entities:")
print(ner(news))