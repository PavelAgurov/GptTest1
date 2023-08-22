"""
    Url processing
"""

url_words : dict[int, int] = {
    "intervals": 6, # Our science
    "science"  : 6  # Our science
}

def get_topic_by_url(url : str) -> int:
    """Check if URL is relevant to some topic"""
    for url_word in url_words.items():
        if url_word[0] in url:
            return url_word[1]
    return None
