def read_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
    

def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks