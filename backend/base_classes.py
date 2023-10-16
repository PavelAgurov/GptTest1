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

    @classmethod
    def Empty(cls):
        """Empty"""
        return TopicScoreItem(None, None, 0, None)

    @classmethod
    def ByFixedPattern(cls, fixed_topic_name : str):
        """Detected by fixed pattern"""
        return TopicScoreItem(-1, fixed_topic_name, 1, 'Detected by fixed pattern')


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
    topics_by_url_info   : str
    leaders_list_str     : str
    senior_leaders_count : bool
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
        """Empty result"""
        return ScoreResultItem(url, input_text_len, 0, '', 0, None, '', None, None, None, False, None)
    
    @classmethod
    def PageNotFound(cls, url : str):
        """Page not found object"""
        return ScoreResultItem(url, 0, 0, '', 0, None, '', None, None, None, False, "Page not found")

    @classmethod
    def Error(cls, url : str, error : str):
        """Error"""
        return ScoreResultItem(url, 0, 0, '', 0, None, 'ERROR', None, None, None, False, error)

    @classmethod
    def Fixed(cls, url : str, main_topics : MainTopics):
        """Fixed topics"""
        return ScoreResultItem(url, 0, 0, '', 0, main_topics, '', None, None, None, False, None)

@dataclass
class FixedTopicPattern:
    """Pattern to fixed topic"""
    url_prefix       : str
    primary_topic    : str
    secondary_topic  : str
    do_not_run_score : bool

    @classmethod
    def Ignore(cls, url : str):
        """Ignore url"""
        return FixedTopicPattern(url, None, None, True)
