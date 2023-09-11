"""
    Topic manager
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from topics import TOPICS_LIST
from backend.base_classes import TopicDefinition


class TopicManager():
    """Topic manager"""

    topic_chunks : list[list[TopicDefinition]]
    topic_dict   : dict[int, str]
    url_words    : dict[str, int]

    def __init__(self):
        self.topic_chunks = [TOPICS_LIST] # utils.grouper(TOPICS_LIST, 4)
        self.topic_dict = {t.id:t.name for t in TOPICS_LIST}
        
        self.url_words = dict[str, int]()
        for topic in TOPICS_LIST:
            if not topic.url_words:
                continue
            for word in topic.url_words:
                self.url_words[word] = topic.id

    def get_topic_chunks(self) -> list[list[TopicDefinition]]:
        """Get topic chunks"""
        return self.topic_chunks
    
    def get_topic_dict(self) -> dict[int, str]:
        """Return topic dict [topic_id, topic_name]"""
        return self.topic_dict
    
    def get_topic_by_url(self, url : str) -> int:
        """Check if URL is relevant to some topic"""
        for url_word_item in self.url_words.items():
            if url_word_item[0] in url:
                return url_word_item[1]
        return None
    
    def get_topic_list(self) -> list[TopicDefinition]:
        """Return all topics"""
        return TOPICS_LIST

    