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
from backend.base_classes import TopicScoreItem, MainTopics, ScoreResultItem, TopicDefinition
from backend.bulk_output import BulkOutput, BulkOutputParams
from backend.html_processors.bs4_processor import get_plain_text_bs4
from backend.gold_data import get_gold_data

PRIORITY_THRESHOLD_ITEM = 0.5

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
    site_map_only    : bool
    skip_translation : bool
    priority_threshold_main : float
    open_api_key     : str
    callbacks        : BackendCallbacks
    footer_texts     : list[str]

@dataclass
class TranslatedResult:
    """Result of translation"""
    lang            : str
    translated_text : str
    no_translation  : bool

@dataclass
class BuildOuputDataResult:
    """Result of buld data"""
    data  : pd.DataFrame
    error : str

class BackEndCore():
    """Main back-end class"""

    backend_params : BackendParams
    llm_manager    : LLMManager
    topic_manager  : TopicManager

    def __init__(self, backend_params : BackendParams):
        self.backend_params = backend_params
        llm_callbacks = LlmCallbacks(
            backend_params.callbacks.report_substatus_callback,
            backend_params.callbacks.used_tokens_callback,
            backend_params.callbacks.report_error_callback
        )
        self.topic_manager = TopicManager()
        self.llm_manager   = LLMManager(backend_params.open_api_key, llm_callbacks)

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

        if self.backend_params.site_map_only:
            return ScoreResultItem.Empty(url)

        topic_dict : dict[int, TopicDefinition] = self.topic_manager.get_topic_dict()

        # load text from URL
        self.report_substatus('Fetch data from URL...')
        input_text = ''
        try:
            input_text = self.fetch_data_from_url(url, read_mode)
        except Exception: # pylint: disable=W0718
            self.report_substatus('Page not found')
            return ScoreResultItem.PageNotFound(url)

        # clean up text for LLM usage
        input_text = self.llm_manager.clean_up_text(input_text)
        input_text_len = len(input_text)
        self.backend_params.callbacks.show_original_text_callback(input_text)
        self.report_substatus(f'Done. Got {input_text_len} chars.')

        # build summary
        self.report_substatus('Build summary...')
        summary = self.llm_manager.refine_text(input_text)
        summary = text_extractor(self.backend_params.footer_texts, summary)
        summary = summary.strip()
        extracted_text_len = len(summary)
        self.report_substatus('Summary is ready')
        self.backend_params.callbacks.show_summary_callback(summary)
        if extracted_text_len == 0:
            return ScoreResultItem(url, input_text_len, 0, '', 0, None, '', None, False)

        # translation (if needed)
        full_translated_text = summary
        translated_lang = ''
        if not self.backend_params.skip_translation:
            translation_result = self.get_translated_text(summary)
            translated_lang = translation_result.lang
            if not translation_result.no_translation and len(translation_result.translated_text) > 0:
                full_translated_text = translation_result.translated_text
        self.backend_params.callbacks.show_extracted_text_callback(full_translated_text)

        self.report_substatus('Run topic score...')
        score_topics_result : ScoreTopicsResult = self.llm_manager.score_topics(
                                                    url,
                                                    full_translated_text,
                                                    self.topic_manager.get_topic_chunks()
                                                )
        self.backend_params.callbacks.show_debug_json_callback(score_topics_result.debug_json_score)
        self.backend_params.callbacks.used_tokens_callback(score_topics_result.used_tokens)
        self.report_substatus('')

        if score_topics_result.error:
            self.backend_params.callbacks.report_error_callback(score_topics_result.error)
            return ScoreResultItem.Error(url, score_topics_result.error)

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

        # topics_score_ordered : dict {topic_index: [score of topic, explanation]}
        topics_score_ordered = collections.OrderedDict(sorted(score_topics_result.result_score.items()))
        
        topics_score_list = list[TopicScoreItem]()
        priority_topics   = list[TopicScoreItem]()
        for score_item in score_topics_result.result_score.items():
            score_item_topic_index = score_item[0]
            score_item_topic_score = topics_score_ordered[score_item_topic_index][0]
            score_item_topic_expln = topics_score_ordered[score_item_topic_index][1]
            if score_item_topic_index not in topic_dict:
                self.backend_params.callbacks.report_error_callback(topics_score_ordered)
                continue
            
            topic_priority = topic_dict[score_item_topic_index].priority
            if topic_priority and topic_priority > 0:
                original_topic_score = score_item_topic_score
                score_item_topic_score = score_item_topic_score * topic_priority
                score_item_topic_expln = f'{score_item_topic_expln} Priority {topic_priority}: {original_topic_score:.2f}=>{score_item_topic_score:.2f}.'
                if score_item_topic_score > PRIORITY_THRESHOLD_ITEM:
                    priority_topics.append(
                        TopicScoreItem(
                            score_item_topic_index,
                            topic_dict[score_item_topic_index].name,
                            score_item_topic_score,
                            score_item_topic_expln
                        ))

            topics_score_list.append(
                TopicScoreItem(
                    topic_dict[score_item_topic_index].id,
                    topic_dict[score_item_topic_index].name,
                    min(score_item_topic_score, 1), 
                    score_item_topic_expln
                )
            )

        # process priority
        if priority_topics:
            priority_topics = sorted(priority_topics, key=lambda x: x.topic_score, reverse=True)
            priority_topic_candidate = priority_topics[0]
            if  priority_topic_candidate.topic_score > main_topics.primary.topic_score and \
                    main_topics.secondary.topic_index != priority_topic_candidate.topic_index and \
                    main_topics.primary.topic_score < self.backend_params.priority_threshold_main:
                main_topics.primary =  priority_topic_candidate
            elif priority_topic_candidate.topic_score > main_topics.secondary.topic_score and \
                    main_topics.primary.topic_index != priority_topic_candidate.topic_index and\
                    main_topics.secondary.topic_score < self.backend_params.priority_threshold_main:
                main_topics.secondary =  priority_topic_candidate

        main_topics.primary.topic_score = min(main_topics.primary.topic_score, 1)
        main_topics.secondary.topic_score = min(main_topics.secondary.topic_score, 1)

        self.backend_params.callbacks.show_main_topics_callback(main_topics)
        self.backend_params.callbacks.show_topics_score_callback(topics_score_list)
        self.report_substatus('')

        score_result_item : ScoreResultItem = ScoreResultItem(
            url,
            input_text_len,
            extracted_text_len,
            translated_lang,
            len(full_translated_text),
            main_topics,
            full_translated_text,
            topics_score_ordered,
            False
        )

        return score_result_item

    def fetch_data_from_url(self, url : str, read_mode : ReadModeHTML) -> str:
        """load text from URL"""
        print('---------------------------------------')
        print(f'Fetch data from URL [{url}]')
        input_text = self.load_html(url, read_mode)
        return input_text

    def get_translated_text(self, text : str) -> TranslatedResult:
        """"Translation"""
        self.report_substatus('Request LLM for translation...')
        translation_result : TranslationResult = self.llm_manager.translate_text(text)
        self.backend_params.callbacks.used_tokens_callback(translation_result.used_tokens)
        translated_lang = translation_result.lang
        if translated_lang in ["English", "en"]:
            self.report_substatus('')
            self.backend_params.callbacks.show_lang_callback("Text is in English. No translation needed.")
            return TranslatedResult("English", text, True)

        if translation_result.error:
            self.report_substatus('')
            self.backend_params.callbacks.show_lang_callback("Translation error.")
            self.backend_params.callbacks.report_error_callback(translation_result.error)
            return TranslatedResult("English", text, True)

        self.backend_params.callbacks.show_lang_callback(f'Language of original text: {translated_lang}')

        if not translation_result.translation:
            self.report_substatus('')
            self.backend_params.callbacks.show_lang_callback("Translation result is empty.")
            return TranslatedResult("English", text, True)

        self.report_substatus('')
        return TranslatedResult(translated_lang, translation_result.translation, False)

    def  get_main_topics(self,
                        topic_dict : dict[str, TopicDefinition],
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
            primary_topic = topic_dict[primary_topic_index].name
            primary_topic_score = result_primary_topic_json['score']
            primary_topic_explanation = result_primary_topic_json['explanation']
        if topic_index_by_url and primary_topic_index != topic_index_by_url: # we have other topic from URL - override
            primary_topic_index = topic_index_by_url
            primary_topic = topic_dict[primary_topic_index].name
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
            secondary_topic = topic_dict[secondary_topic_index].name
            secondary_topic_score = result_secondary_topic_json['score']
            secondary_topic_explanation = result_secondary_topic_json['explanation']

        # if now we have primary the same as secondary, because owerride it by URL-topic - get primary as secondary
        if primary_topic_index == secondary_topic_index and result_primary_topic_json and primary_topic_is_url:
            secondary_topic_index = result_primary_topic_json['topic_id']
            secondary_topic = topic_dict[secondary_topic_index].name
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
