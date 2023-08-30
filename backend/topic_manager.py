"""
    Topic manager
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from topics import TOPICS_LIST

url_words : dict[int, int] = {
    "intervals": 6, # Our science
    "science"  : 6,
    "job-opportunities": 11, # Job
    "job-remotely": 11,
   "job-details": 11,
   "job-interview": 11,
}


class TopicManager():
    """Topic manager"""

    topic_chunks : any
    topic_dict   : dict[int, str]

    def __init__(self):
        self.topic_chunks = [TOPICS_LIST] # utils.grouper(TOPICS_LIST, 4)
        self.topic_dict = {t[0]:t[1] for t in TOPICS_LIST}

    def get_topic_chunks(self) -> any:
        """Get topic chunks"""
        return self.topic_chunks
    
    def get_topic_dict(self) -> dict[int, str]:
        """Return topic dict [topic_id, topic_name]"""
        return self.topic_dict
    
    def get_topic_by_url(self, url : str) -> int:
        """Check if URL is relevant to some topic"""
        for url_word in url_words.items():
            if url_word[0] in url:
                return url_word[1]
        return None
    
    def  get_topic_list(self) -> any:
        """Return all topics"""
        return TOPICS_LIST

    