"""
    Back-end
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903,W1203

import re
from enum import Enum
from dataclasses import dataclass
import collections
from unstructured.partition.html import partition_html
import urllib
from urllib.parse import urlparse
import pandas as pd
import logging

from backend.llm_manager import LLMManager, LlmCallbacks, ScoreTopicsResult, TranslationResult, LeadersListResult
from backend.topic_manager import TopicManager, TopicDetectedByURL
from backend.base_classes import TopicScoreItem, MainTopics, ScoreResultItem, TopicDefinition, FixedTopicPattern
from backend.bulk_output import BulkOutput, BulkOutputParams
from backend.html_processors.bs4_processor import get_plain_text_bs4
from backend.gold_data import get_gold_data
from backend.tuning_manager import TuningManager
from data.parser_html_classes import HTML_CLASSES_WHITELIST, HTML_CLASSES_BLACKLIST, EXCLUDED_SENTENSES
from data.fixed_topic_patterns import FIXED_TOPIC_PATTERNS_DICT

logger : logging.Logger = logging.getLogger()

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
    site_map_only      : bool
    skip_translation   : bool
    override_by_url_words      : bool
    override_by_url_words_less : float
    url_words_add       : float
    skip_summary       : bool
    use_topic_priority : bool
    use_leaders        : bool
    all_secrets        : dict[str, any]
    open_api_key_ui    : str
    callbacks          : BackendCallbacks

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

@dataclass
class TuningPrompt:
    """Tuning prompt str"""
    prompt : str
    tokens : int

class BackEndCore():
    """Main back-end class"""

    backend_params : BackendParams
    llm_manager    : LLMManager
    topic_manager  : TopicManager
    tuning_manager : TuningManager

    PMI_COMPANY_NAMES = ['pmi', 'philip morris international', 'philip morris international (pmi)']
    AUTHOR_PATTERN    = 'Written by'

    def __init__(self, backend_params : BackendParams):
        self.backend_params = backend_params
        llm_callbacks = LlmCallbacks(
            backend_params.callbacks.report_substatus_callback,
            backend_params.callbacks.used_tokens_callback,
            backend_params.callbacks.report_error_callback
        )
        self.topic_manager  = TopicManager()
        self.llm_manager    = LLMManager(backend_params.all_secrets, backend_params.open_api_key_ui, llm_callbacks)
        self.tuning_manager = TuningManager()

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
            domain = urlparse(url).netloc

            html_classes_whitelist = HTML_CLASSES_WHITELIST.get(domain, None)
            html_classes_blacklist = HTML_CLASSES_BLACKLIST.get(domain, None)

            with urllib.request.urlopen(url) as f:
                html = f.read()
            result2 = get_plain_text_bs4(html, html_classes_whitelist, html_classes_blacklist)

        if loading_mode_bs == ReadModeHTML.PARTITION.value:
            return result1
        if loading_mode_bs == ReadModeHTML.BS4.value:
            return result2

        # mixed mode - return where we have more text
        if len(result1) > len(result2):
            return result1
        return result2

    def run(self, url_list : list[str], read_mode : ReadModeHTML, only_read_html : bool) -> list[ScoreResultItem]:
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
            result_data.append(self.run_one(url, read_mode, only_read_html))
            if len(url_list) > 1:
                self.report_substatus('')

        self.report_status('Done')
        return result_data

    def run_one(self, url : str, read_mode : ReadModeHTML, only_read_html : bool) -> ScoreResultItem:
        """"Run process for one URL"""

        if self.backend_params.site_map_only:
            return ScoreResultItem.Empty(url)

        topic_dict : dict[int, TopicDefinition] = self.topic_manager.get_topic_dict()

        fixed_topic_pattern : FixedTopicPattern = None
        for prefix_item in FIXED_TOPIC_PATTERNS_DICT.items():
            if url.startswith(prefix_item[0]):
                fixed_topic_pattern = prefix_item[1]
                break

        # we found pattern and it's not needed to run score for it
        if fixed_topic_pattern and fixed_topic_pattern.do_not_run_score:
            fixed_primary_topic_name   = fixed_topic_pattern.primary_topic
            fixed_secondary_topic_name = fixed_topic_pattern.secondary_topic

            if not fixed_primary_topic_name and not fixed_secondary_topic_name:
                logger.error(f'Page was ignored {url}')
                return ScoreResultItem.Error(url, "Ignored")

            fixed_primary = TopicScoreItem.Empty()
            if fixed_primary_topic_name:
                fixed_primary = TopicScoreItem(-1, fixed_primary_topic_name, 1, 'Detected by fixed pattern')
            
            fixed_secondary = TopicScoreItem.Empty()
            if fixed_secondary_topic_name:
                fixed_secondary = TopicScoreItem(-1, fixed_secondary_topic_name, 1, 'Detected by fixed pattern')

            fixed_main_topics = MainTopics(fixed_primary, fixed_secondary)

            self.backend_params.callbacks.show_main_topics_callback(fixed_main_topics)
            self.report_substatus('')

            return ScoreResultItem.Fixed(url, fixed_main_topics)

        # load text from URL
        self.report_substatus('Fetch data from URL...')
        input_text = ''
        try:
            input_text = self.fetch_data_from_url(url, read_mode)
        except Exception: # pylint: disable=W0718
            self.report_substatus('Page not found')
            logger.error(f'Page not found {url}')
            return ScoreResultItem.PageNotFound(url)

        # clean up text for LLM usage
        input_text = self.llm_manager.clean_up_text(input_text)
        input_text_len = len(input_text)
        self.backend_params.callbacks.show_original_text_callback(input_text)
        self.report_substatus(f'Done. Got {input_text_len} chars.')

        if only_read_html:
            logger.debug('Flag only_read_html is True. Exit.')
            return ScoreResultItem.Empty(url, input_text_len)

        if input_text_len == 0:
            self.report_substatus('Input is empty')
            logger.error(f'Input is empty {url}')
            return ScoreResultItem.Empty(url, input_text_len)

        # detec leaders
        senior_pmi_leaders = []
        leaders_list_str = None
        self.report_substatus('Detect Leaders...')
        logger.info('Detect Leaders...')
        leaders_list : LeadersListResult = self.llm_manager.detect_leaders(url, input_text)
        logger.debug(leaders_list)
        if leaders_list and leaders_list.leaders:
            for leader in leaders_list.leaders: # not the best hack to detect references without company name
                if leader.company in leader.title:
                    leader.company = 'PMI'
            leaders_list_str = '|'.join([f'{leader.name}, {leader.company}, {leader.title}, {leader.senior}[{leader.counter}]' for leader in leaders_list.leaders if leader.name])
            senior_pmi_leaders = [
                    leader for leader in leaders_list.leaders 
                    if leader.senior and # only senior staff
                        leader.company and leader.company.lower() in self.PMI_COMPANY_NAMES # only staff in defined companies
                ]
        self.report_substatus('')

        # topic detected by URL, sorted by detected position
        topic_index_by_url_list : list[TopicDetectedByURL] = self.topic_manager.get_topics_by_url(url)
        logger.debug(f'topic_index_by_url_list={topic_index_by_url_list}')
        topics_by_url_info = ','.join([f'{topic_dict[t.topic_index].name}[{t.url_position}]' for t in topic_index_by_url_list])

        # build summary
        summary = input_text
        if not self.backend_params.skip_summary:
            self.report_substatus('Build summary...')
            summary = self.llm_manager.refine_text(input_text)
            summary = summary.strip()
            self.report_substatus('Summary is ready')
            self.backend_params.callbacks.show_summary_callback(summary)

        topics_score_list    = []
        translated_lang      = None
        full_translated_text = ''
        topics_score_ordered = None
        score_topics_result : ScoreTopicsResult

        extracted_text_len = len(summary)
        if extracted_text_len > 0:
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
            score_topics_result = self.llm_manager.score_topics(
                                                        url,
                                                        full_translated_text,
                                                        self.topic_manager.get_topic_list()
                                                    )
            self.backend_params.callbacks.show_debug_json_callback(score_topics_result.debug_json_score)
            self.backend_params.callbacks.used_tokens_callback(score_topics_result.used_tokens)
            self.report_substatus('')
        else:
            logger.warning(f'Summary is empty {url}')
            empty_result_score = {t[0]: (0, None) for t in topic_dict.items()}
            score_topics_result = ScoreTopicsResult(0, None, None, empty_result_score)

        if score_topics_result.error:
            self.backend_params.callbacks.report_error_callback(score_topics_result.error)
            return ScoreResultItem.Error(url, score_topics_result.error)

        self.report_substatus('Calculate scores...')
        
        # topics_score_ordered : dict {topic_index: [score of topic, explanation]}
        topics_score_ordered = collections.OrderedDict(sorted(score_topics_result.result_score.items()))

        topics_score_list = list[TopicScoreItem]()
        for score_item in score_topics_result.result_score.items():
            score_item_topic_index = score_item[0]
            score_item_topic_score = topics_score_ordered[score_item_topic_index][0]
            score_item_topic_expln = topics_score_ordered[score_item_topic_index][1]
            if score_item_topic_index not in topic_dict:
                self.backend_params.callbacks.report_error_callback(f'Unknown topic index. JSON: {topics_score_ordered}. URL: {url}')
                continue
            
            if self.backend_params.url_words_add > 0:
                for topic_index_by_url in topic_index_by_url_list:
                    if  topic_index_by_url.topic_index == score_item_topic_index:
                        # we can add url score if detected score > 0 OR we have no input for detection
                        if score_item_topic_score > 0 or extracted_text_len == 0:
                            original_topic_score = score_item_topic_score
                            score_item_topic_score = score_item_topic_score + self.backend_params.url_words_add
                            score_item_topic_expln = f'!{score_item_topic_expln}. Detected by URL: {original_topic_score:.2f}=>{score_item_topic_score:.2f}'

            if self.backend_params.use_topic_priority:
                topic_priority = topic_dict[score_item_topic_index].priority
                if topic_priority and topic_priority > 0 and score_item_topic_score > 0:
                    original_topic_score = score_item_topic_score
                    score_item_topic_score = score_item_topic_score * topic_priority
                    score_item_topic_expln = f'*{score_item_topic_expln} Priority {topic_priority}: {original_topic_score:.2f}=>{score_item_topic_score:.2f}.'

            topics_score_list.append(
                TopicScoreItem(
                    topic_dict[score_item_topic_index].id,
                    topic_dict[score_item_topic_index].name,
                    score_item_topic_score,
                    score_item_topic_expln
                )
            )

        topics_score_list = sorted(topics_score_list, key=lambda t: t.topic_score, reverse=True)

        self.report_substatus('Calculate primary and secondary topics...')

        primary_item = topics_score_list[0]
        primary_main_topic = TopicScoreItem(
            primary_item.topic_index,
            primary_item.topic,
            primary_item.topic_score,
            primary_item.explanation
        )
        secondary_item = topics_score_list[1]
        secondary_main_topic = TopicScoreItem(
            secondary_item.topic_index,
            secondary_item.topic,
            secondary_item.topic_score,
            secondary_item.explanation
        )

        if self.backend_params.override_by_url_words and len(topic_index_by_url_list) > 0:
            topic_index_by_url = topic_index_by_url_list[0].topic_index
            if primary_main_topic.topic_index != topic_index_by_url and primary_main_topic.topic_score <= self.backend_params.override_by_url_words_less:
                secondary_main_topic = primary_main_topic
                primary_main_topic = TopicScoreItem(
                    topic_index_by_url,
                    topic_dict[topic_index_by_url].name,
                    1,
                    'Detected by URL'
                )

        senior_pmi_leaders_counter = len(senior_pmi_leaders)
        logger.info(f'senior_pmi_leaders_counter={senior_pmi_leaders_counter}')

        if self.backend_params.use_leaders and senior_pmi_leaders_counter >= 2:
            secondary_main_topic = primary_main_topic
            primary_main_topic = TopicScoreItem(
                -1,
                'Leadership content',
                1,
                'Detected by Leader extractor'
            )

        if self.backend_params.use_leaders and len(senior_pmi_leaders) > 0:
            for pmi_leader in senior_pmi_leaders:
                author_pattern = rf'{self.AUTHOR_PATTERN}\s+{pmi_leader.name}'
                res = re.findall(author_pattern, input_text)
                if res:
                    secondary_main_topic = primary_main_topic
                    primary_main_topic = TopicScoreItem(
                        -1,
                        'Leadership content',
                        1,
                        'Detected by author'
                    )
                    break

        # we have fixed pattern - overwrite all
        if fixed_topic_pattern:
            if fixed_topic_pattern.primary_topic:
                secondary_main_topic = primary_main_topic
                primary_main_topic   = TopicScoreItem(-1, fixed_topic_pattern.primary_topic, 1, 'Detected by fixed pattern')
            if fixed_topic_pattern.secondary_topic:
                secondary_main_topic = TopicScoreItem(-1, fixed_topic_pattern.secondary_topic, 1, 'Detected by fixed pattern')

        main_topics = MainTopics(primary_main_topic, secondary_main_topic)
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
            topics_by_url_info,
            leaders_list_str,
            senior_pmi_leaders_counter,
            False
        )

        return score_result_item

    def fetch_data_from_url(self, url : str, read_mode : ReadModeHTML) -> str:
        """load text from URL"""
        logger.info('---------------------------------------')
        logger.info(f'Fetch data from URL [{url}] read_mode={read_mode}')
        input_text = self.load_html(url, read_mode)

        if EXCLUDED_SENTENSES and input_text:
            for sentense in EXCLUDED_SENTENSES:
                input_text = input_text.replace(sentense, ' ')

        if input_text:
            input_text = input_text.strip()

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

    def build_ouput_data(self, bulk_result : list[ScoreResultItem], bulk_output_params : BulkOutputParams) -> BuildOuputDataResult:
        """Build output data frame"""
        topic_list = self.topic_manager.get_topic_list()
        gold_data  = None
        if bulk_output_params.inc_golden_data:
            gold_data = get_gold_data()
        
        error = None
        if gold_data and gold_data.error:
            error = gold_data.error

        data = BulkOutput().create_data(topic_list, bulk_result, gold_data, bulk_output_params)
        return BuildOuputDataResult(data, error)

    def get_tuning_prompt(self, bulk_data : pd.DataFrame) -> TuningPrompt:
        """Get tuning prompt"""
        topic_list = self.topic_manager.get_topic_list()
        tuning_prompt = self.tuning_manager.get_tuning_prompt(bulk_data, topic_list)
        tokens_in_prompt = self.llm_manager.get_token_count(tuning_prompt)
        return TuningPrompt(tuning_prompt, tokens_in_prompt)
    
    def run_tuning_prompt(self, tuning_prompt : str) -> str:
        """Get tuning result"""
        return len(tuning_prompt)