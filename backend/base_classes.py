"""
    Classes
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class TopicDefinition:
    """Topic definision"""
    id : int
    name : str
    description : str
    url_words : list[str] = None

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
    page_not_found       : bool
