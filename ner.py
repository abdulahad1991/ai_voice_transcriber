from transformers import pipeline

# This will redownload everything from scratch
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

print("✅ NER model downloaded successfully!")
