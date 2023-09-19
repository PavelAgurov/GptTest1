"""
    Back-end
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from enum import Enum
from dataclasses import dataclass
import collections
from unstructured.partition.html import partition_html
import urllib
import pandas as pd

from backend.llm_manager import LLMManager, LlmCallbacks, ScoreTopicsResult, TranslationResult
from backend.text_processing import text_extractor
from backend.topic_manager import TopicManager
from backend.base_classes import TopicScoreItem, MainTopics, ScoreResultItem
from backend.bulk_output import BulkOutput, BulkOutputParams
from backend.html_processors.bs4_processor import get_plain_text_bs4
from backend.gold_data import get_gold_data

class ReadModeHTML(Enum):
    """Types of HTML reading"""
    PARTITION = 'By unstructured.partition'
    BS4       = 'By bs4'
    MIXED     = 'Mixed'

@dataclass
class BackendCallbacks:
    """Set of callbacks"""
    report_status_callback       : any
    report_substatus_callback    : any
    used_tokens_callback         : any
    show_original_text_callback  : any
    report_error_callback        : any
    show_summary_callback        : any
    show_lang_callback           : any
    show_extracted_text_callback : any
    show_debug_json_callback     : any
    show_main_topics_callback    : any
    show_topics_score_callback   : any

@dataclass
class BackendParams:
    """Backend params"""
    open_api_key : str
    callbacks : BackendCallbacks
    score_by_summary : bool
    footer_texts : list[str]

@dataclass
class TranslatedParagraphs:
    """Result of translation"""
    lang : str
    paragraphs : list[str]

@dataclass
class BuildOuputDataResult:
    """Result of buld data"""
    data  : pd.DataFrame
    error : str

class BackEndCore():
    """Main back-end class"""

    backend_params : BackendParams
    llm : LLMManager
    topic_manager : TopicManager

    def __init__(self, backend_params : BackendParams):
        self.backend_params = backend_params
        llm_callbacks = LlmCallbacks(
            backend_params.callbacks.report_substatus_callback,
            backend_params.callbacks.used_tokens_callback,
            backend_params.callbacks.report_error_callback
        )
        self.topic_manager = TopicManager()
        self.llm = LLMManager(backend_params.open_api_key, llm_callbacks)

    def report_status(self, status_str : str):
        """Report first line of status"""
        self.backend_params.callbacks.report_status_callback(status_str)

    def report_substatus(self, substatus_str : str):
        """Report second line of status"""
        self.backend_params.callbacks.report_substatus_callback(substatus_str)

    def load_html(self, url : str, loading_mode_bs : ReadModeHTML) -> str:
        """Load HTML"""
        # based on partition_html
        result1 = ''
        if loading_mode_bs in [ReadModeHTML.PARTITION.value, ReadModeHTML.MIXED.value]:
            elements = partition_html(url = url)
            result1 = "\n\n".join([str(el).strip() for el in elements])

        # based on BS4
        result2 = ''
        if loading_mode_bs in [ReadModeHTML.BS4.value, ReadModeHTML.MIXED.value]:
            with urllib.request.urlopen(url) as f:
                html = f.read()
            result2 = get_plain_text_bs4(html)

        if loading_mode_bs == ReadModeHTML.PARTITION.value:
            return result1
        if loading_mode_bs == ReadModeHTML.BS4.value:
            return result2

        # mixed mode - return where we have more text
        if len(result1) > len(result2):
            return result1
        return result2

    def run(self, url_list : list[str], read_mode : ReadModeHTML) -> list[ScoreResultItem]:
        """Run process for bulk URLs"""

        result_data = list[ScoreResultItem]()
        for index, url in enumerate(url_list):
            url = url.strip()
            if not url:
                continue
            if len(url_list) > 1:
                self.report_status(f'Processing URL [url: {index+1}/{len(url_list)}] {url}...')
            else:
                self.report_status(f'Processing URL {url}...')
            result_data.append(self.run_one(url, read_mode))
            if len(url_list) > 1:
                self.report_substatus('')

        self.report_status('Done')
        return result_data

    def run_one(self, url : str, read_mode : ReadModeHTML) -> ScoreResultItem:
        """"Run process for one URL"""

        topic_dict : dict[int, str] = self.topic_manager.get_topic_dict()

        # load text from URL
        self.report_substatus('Fetch data from URL...')
        print('---------------------------------------')
        print(f'Fetch data from URL [{url}]')
        
        input_text = ''
        input_text_len = 0
        try:
            input_text = self.load_html(url, read_mode)
            input_text_len = len(input_text)
            self.report_substatus(f'Done. Got {input_text_len} chars.')
        except Exception: # pylint: disable=W0718
            self.report_substatus('Page not found')
            return ScoreResultItem(url, 0, 0, 0, 0,
                None,
                None,
                None,
                True
            )

        # clean up for LLM
        input_text = self.llm.clean_up(input_text)
        self.backend_params.callbacks.show_original_text_callback(input_text)

        # slit into paragraphs (by summary or by original text)
        paragraph_list = self.get_paragraph_list(self.backend_params.score_by_summary, input_text)
        extracted_text_len = len('\n'.join(paragraph_list))

        # translation (if needed)
        translated_paragraphs = self.get_translated_paragraph_list(paragraph_list)
        translated_paragraph_list = translated_paragraphs.paragraphs
        if translated_paragraph_list:
            full_translated_text = '\n'.join(translated_paragraph_list)
        else:
            full_translated_text = 'Error during translation'
        self.backend_params.callbacks.show_extracted_text_callback(full_translated_text)

        self.report_substatus('Run topic score...')
        score_topics_result : ScoreTopicsResult = self.llm.score_topics(
                                                    url,
                                                    translated_paragraph_list,
                                                    self.topic_manager.get_topic_chunks()
                                                )
        self.backend_params.callbacks.show_debug_json_callback(score_topics_result.debug_json_score)
        self.backend_params.callbacks.used_tokens_callback(score_topics_result.used_tokens)
        self.report_substatus('')

        if score_topics_result.error:
            self.backend_params.callbacks.report_error_callback(score_topics_result.error)
            return []

        self.report_substatus('Calculate primary and secondary topics...')
        topic_index_by_url : int = self.topic_manager.get_topic_by_url(url)
        result_primary_topic_json = score_topics_result.primary_topic_json
        result_secondary_topic_json = score_topics_result.secondary_topic_json
    
        main_topics = self.get_main_topics(
                topic_dict,
                topic_index_by_url,
                result_primary_topic_json, 
                result_secondary_topic_json
        )
        self.backend_params.callbacks.show_main_topics_callback(main_topics)
        self.report_substatus('')

        topics_score_ordered = collections.OrderedDict(sorted(score_topics_result.result_score.items()))
        topics_score_list = []
        for score_item in score_topics_result.result_score.items():
            score_item_topic_index = score_item[0]
            if score_item_topic_index in topic_dict:
                topics_score_list.append([topic_dict[score_item_topic_index], *topics_score_ordered[score_item_topic_index]])
            else:
                self.backend_params.callbacks.report_error_callback(topics_score_ordered)
        self.backend_params.callbacks.show_topics_score_callback(topics_score_list)

        score_result_item : ScoreResultItem = ScoreResultItem(
            url,
            input_text_len,
            extracted_text_len,
            translated_paragraphs.lang, 
            len(full_translated_text),
            main_topics,
            full_translated_text,
            topics_score_ordered,
            False
        )

        return score_result_item

    def get_paragraph_list(self, by_summary : bool, input_text : str) -> []:
        """"Get paragpaths (as summary or 'as is')"""
        if by_summary:
            summary = self.llm.refine_text(input_text)
            summary = text_extractor(self.backend_params.footer_texts, summary)
        else:
            summary = text_extractor(self.backend_params.footer_texts, input_text)

        if not summary:
            return []
        self.report_substatus('Summary is ready')
        self.backend_params.callbacks.show_summary_callback(summary)

        if not by_summary:
            paragraph_list = self.llm.split_text_to_paragraphs(summary)
        else:
            paragraph_list = [summary]
        
        return paragraph_list

    def get_translated_paragraph_list(self, paragraph_list : list[str]) -> TranslatedParagraphs:
        """"Translation"""
        translated_list = []
        translated_lang = "None"
        no_translation = False

        for i, paragraph_text in enumerate(paragraph_list):
            self.report_substatus(f'Request LLM for translation paragraph: {i+1}/{len(paragraph_list)}...')
            translation_result : TranslationResult = self.llm.translate_text(paragraph_text)
            self.backend_params.callbacks.used_tokens_callback(translation_result.used_tokens)
            translated_lang = translation_result.lang

            if translated_lang in ["English", "en"]:
                no_translation = True
                break

            if translation_result.error:
                no_translation = True
                self.backend_params.callbacks.report_error_callback(translation_result.error)
                break

            self.backend_params.callbacks.show_lang_callback(f'Language of original text: {translated_lang}')

            translated_text = translation_result.translation
            if translated_text:
                translated_list.append(translated_text)
            else:
                no_translation = True
                self.backend_params.callbacks.report_error_callback('Translation error')
                break

        if no_translation:
            self.backend_params.callbacks.show_lang_callback("Text is in English. No translation needed.")
            translated_list = paragraph_list # just use "as is"

        self.report_substatus('')
        return TranslatedParagraphs(translated_lang, translated_list)

    def  get_main_topics(self,
                        topic_dict : dict,
                        topic_index_by_url : int,
                        result_primary_topic_json : any,
                        result_secondary_topic_json : any) -> MainTopics:
        """Define main topics - primary and secondary"""

        # logic is a bit complicated here:
        # - depect primary topic from LLM
        # - if we have different primary topic detected from URL - assign it as primary, skipp LLM version
        #   (but if primary topic from URL is the same as from LLM - save LLM version)
        # - if secondary topic is equal to primary now - try to re-assign LLM primary into secondary

        primary_topic_index = -1
        primary_topic = ""
        primary_topic_score = 0
        primary_topic_explanation =""
        primary_topic_is_url = False
        if result_primary_topic_json: # we have primary topic from LLM
            primary_topic_index = result_primary_topic_json['topic_id']
            primary_topic = topic_dict[primary_topic_index]
            primary_topic_score = result_primary_topic_json['score']
            primary_topic_explanation = result_primary_topic_json['explanation']
        if topic_index_by_url and primary_topic_index != topic_index_by_url: # we have other topic from URL - override
            primary_topic_index = topic_index_by_url
            primary_topic = topic_dict[primary_topic_index]
            primary_topic_score = 1
            primary_topic_explanation = "Detected from URL"
            primary_topic_is_url = True

        # secondary topic
        secondary_topic_index = -1
        secondary_topic = ""
        secondary_topic_score = 0
        secondary_topic_explanation = ""
        if result_secondary_topic_json: # secondary topic from LLM
            secondary_topic_index = result_secondary_topic_json['topic_id']
            secondary_topic = topic_dict[secondary_topic_index]
            secondary_topic_score = result_secondary_topic_json['score']
            secondary_topic_explanation = result_secondary_topic_json['explanation']

        # if now we have primary the same as secondary, because owerride it by URL-topic - get primary as secondary
        if primary_topic_index == secondary_topic_index and result_primary_topic_json and primary_topic_is_url:
            secondary_topic_index = result_primary_topic_json['topic_id']
            secondary_topic = topic_dict[secondary_topic_index]
            secondary_topic_score = result_primary_topic_json['score']
            secondary_topic_explanation = result_primary_topic_json['explanation']

        result = MainTopics(
            primary = TopicScoreItem(
                topic_index = primary_topic_index,
                topic       = primary_topic,
                topic_score = primary_topic_score,
                explanation = primary_topic_explanation
            ),
            secondary = TopicScoreItem(
                topic_index = secondary_topic_index,
                topic       = secondary_topic,
                topic_score = secondary_topic_score,
                explanation = secondary_topic_explanation
            )
        )

        return result

    def build_ouput_data(self, bulk_result : list[ScoreResultItem], bulk_output_params : BulkOutputParams) -> BuildOuputDataResult:
        """Build output data frame"""
        topic_list = self.topic_manager.get_topic_list()
        gold_data  = None
        if bulk_output_params.inc_gold_data:
            gold_data = get_gold_data()
        
        error = None
        if gold_data and gold_data.error:
            error = gold_data.error

        data = BulkOutput().create_data(topic_list, bulk_result, gold_data, bulk_output_params)
        return BuildOuputDataResult(data, error)
