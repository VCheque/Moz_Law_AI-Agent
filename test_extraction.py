from extract_texts import extract_texts_from_pdfs

texts = extract_texts_from_pdfs("data")

print(f"{len(texts)} documents extracted successfully.\n")

# Show an example
example = texts[0]
print("Filename:", example["filename"])
print("Category:", example["category"])
print("Text preview:\n")
print(example["text"][:1000])  # First 1000 characters
