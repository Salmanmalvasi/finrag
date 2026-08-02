def chunk_text(text, chunk_size=200, overlap=50):
    """
    Split text into overlapping word chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    sample_text = """
    Apple reported record revenue this quarter.
    Services revenue increased significantly.
    Management expects continued growth driven by AI investments.
    The company also expanded manufacturing capacity in Asia.
    """

    chunks = chunk_text(sample_text, chunk_size=8, overlap=2)

    for i, chunk in enumerate(chunks):

        print(f"Chunk {i+1}")
        print(chunk)
        print("-" * 40)