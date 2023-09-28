"""
    Classes
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional

@dataclass_json
@dataclass
class TopicDefinition:
    """Topic definision"""
    id : int
    name : str
    description : str
    url_words : Optional[list[str]] = None
    priority  : Optional[int] = 0

    def get_url_words_str(self) -> str:
        """Get url words as string"""
        if not self.url_words:
            return ''
        return ';'.join(self.url_words)

@dataclass_json
@dataclass
class TopicList:
    """List of topics"""
    topics : list[TopicDefinition]

@dataclass
class TopicScoreItem:
    """Topic score item"""
    topic_index : int
    topic       : str
    topic_score : float
    explanation : str

@dataclass
class MainTopics:
    """Primary and secondary topics"""
    primary   : TopicScoreItem
    secondary : TopicScoreItem

@dataclass
class ScoreResultItem:
    """Result of score of URL"""
    current_url          : str
    input_text_len       : int
    extracted_text_len   : int
    translated_lang      : str
    transpated_text_len  : int
    main_topics          : MainTopics
    full_translated_text : str
    ordered_result_score : list
    error                : str

    def get_main_topic_primary_item(self) -> TopicScoreItem:
        """Get primary topic if exists"""
        if not self.main_topics:
            return None
        return self.main_topics.primary

    def get_main_topic_secondary_item(self) -> TopicScoreItem:
        """Get secondary topic if exists"""
        if not self.main_topics:
            return None
        return self.main_topics.secondary

    def get_main_topic_primary(self) -> str:
        """Get primary topic if exists"""
        if not self.main_topics or not self.main_topics.primary:
            return ''
        return self.main_topics.primary.topic

    def get_main_topic_secondary(self) -> str:
        """Get secondary topic if exists"""
        if not self.main_topics or not self.main_topics.secondary:
            return ''
        return self.main_topics.secondary.topic

    @classmethod
    def Empty(cls, url : str, input_text_len : int = 0):
        """Empry result"""
        return ScoreResultItem(url, input_text_len, 0, '', 0, None, '', None, None)
    
    @classmethod
    def PageNotFound(cls, url : str):
        """Page not found object"""
        return ScoreResultItem(url, 0, 0, '', 0, None, '', None, "Page not found")

    @classmethod
    def Error(cls, url : str, error : str):
        """Error"""
        return ScoreResultItem(url, 0, 0, '', 0, None, 'ERROR', None, error)

