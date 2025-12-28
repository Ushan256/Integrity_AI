import re

def preprocess_text(text):
    """
    Cleans and tokenizes input text.
    This is the foundation for feature extraction and ML.
    """

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Split text into words (tokens)
    words = text.split()

    return words


# ===========================
# TEST THE FUNCTION
# ===========================
if __name__ == "__main__":
    sample_text = (
        "I completed the assignment yesterday, and it was quite challenging. "
        "However, I learned a lot from the process!"
    )

    tokens = preprocess_text(sample_text)

    print("Original Text:")
    print(sample_text)
    print("\nProcessed Tokens:")
    print(tokens)
