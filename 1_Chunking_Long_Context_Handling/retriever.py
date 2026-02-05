def get_relevant_chunk(chunks, question):
    print(chunks,question)
    question_words = set(question.lower().split())
    best_chunk = None
    best_score = 0

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score

def retrieve_chunk(chunks, question, min_score=2):
    chunk, score = get_relevant_chunk(chunks, question)
    print("score",score)
    if score < min_score:
        return None

    return chunk