"""
    Topic manager
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903
import os
import copy
from dataclasses import dataclass

from data.topics import TOPICS_LIST
from backend.base_classes import TopicDefinition, TopicList


@dataclass
class TopicPriority:
    """To store topic priority"""
    topic_index    : int
    topic_priority : int

class TopicManager():
    """Topic manager"""

    topic_list : list[TopicDefinition]
    topic_dict : dict[int, TopicDefinition]
    url_words  : dict[str, TopicPriority]

    __DISK_FOLDER = '.topics'
    __TOPIC_FILE = 'topics.json'

    def __init__(self):
        self.topic_dict = {t.id:t for t in TOPICS_LIST}
        
        self.url_words = dict[str, int]()
        for topic in TOPICS_LIST:
            if not topic.url_words:
                continue
            for word in topic.url_words:
                self.url_words[word] = TopicPriority(topic.id, topic.priority)

        os.makedirs(self.__DISK_FOLDER, exist_ok=True)

        if not self.load_from_disk():
            self.topic_list = TOPICS_LIST

    def get_topic_list(self) -> list[TopicDefinition]:
        """Get topic list"""
        return self.topic_list
    
    def get_topic_dict(self) -> dict[int, TopicDefinition]:
        """Return topic dict [topic_id, TopicDefinition]"""
        return self.topic_dict
    
    def get_topics_by_url(self, url : str) -> list[TopicPriority]:
        """Check if URL is relevant to some topic"""
        detected_list = list[TopicPriority]()
        for url_word_item in self.url_words.items():
            if url_word_item[0] in url:
                detected_list.append(url_word_item[1])
        if detected_list:
            detected_list = sorted(detected_list, key=lambda x: x.topic_priority, reverse=True)
        return detected_list
   
    def save_topic_descriptions(self, updated_list: list[TopicDefinition]):
        """Save updated descriptions"""
        new_descriptions = {t.id:t.description for t in updated_list}
        new_copy = copy.deepcopy(TOPICS_LIST)
        for topic in new_copy:
            if topic.id in new_descriptions:
                topic.description = new_descriptions[topic.id]
        self.topic_list = new_copy

        topic_list = TopicList(new_copy)
        json_str = topic_list.to_json(indent=4)  # pylint: disable=E1101
        file_name = os.path.join(self.__DISK_FOLDER, self.__TOPIC_FILE)
        with open(file_name, "wt", encoding="utf-8") as f:
            f.write(json_str)

    def load_from_disk(self) -> bool:
        """Load from disk"""
        file_name = os.path.join(self.__DISK_FOLDER, self.__TOPIC_FILE)
        if not os.path.isfile(file_name):
            return  False
        with open(file_name, "rt", encoding="utf-8") as f:
            json_str= f.read()
        topic_list : TopicList = TopicList.from_json(json_str) # pylint: disable=E1101
        self.topic_list = topic_list.topics
        return True

    def reset_all_topics(self):
        """Reset all topics"""
        file_name = os.path.join(self.__DISK_FOLDER, self.__TOPIC_FILE)
        if not os.path.isfile(file_name):
            return
        os.remove(file_name)
        self.topic_list = copy.deepcopy(TOPICS_LIST)

