from docx import Document

doc = Document("F:\word.docx")

print(doc)
for para in doc.paragraphs:
    print(para.text)

