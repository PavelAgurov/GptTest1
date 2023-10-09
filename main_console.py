"""
    Console app
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,W1203

import os
import logging

from backend.backend_core import BackEndCore, BackendParams, BackendCallbacks, ReadModeHTML
from backend.base_classes import ScoreResultItem, MainTopics, TopicScoreItem
from utils.app_logger import init_root_logger

total_used_tokens = 0
logger : logging.Logger

def report_status(status_str : str):
    """Show first status line"""
    if status_str:
        logger.info(status_str)

def report_substatus(substatus_str : str):
    """Show second line of status"""
    if substatus_str:
        logger.info(substatus_str)

def used_tokens_callback(used_tokens : int):
    """Update token counter"""
    global total_used_tokens # pylint: disable=W0603
    total_used_tokens += used_tokens

def show_original_text_callback(text : str): # pylint: disable=W0613
    """Show original text"""
    pass  # pylint: disable=W0107

def report_error_callback(err : str):
    """Report error"""
    logger.error(f'ERROR {err}')

def show_summary_callback(summary : str): # pylint: disable=W0613
    """Show summary"""
    pass  # pylint: disable=W0107

def show_lang_callback(lang_str : str): # pylint: disable=W0613
    """Show lang string"""
    pass  # pylint: disable=W0107

def show_extracted_text_callback(text : str): # pylint: disable=W0613
    """Show extracted text"""
    pass  # pylint: disable=W0107

def show_debug_json_callback(json_str : str): # pylint: disable=W0613
    """Show debug json"""
    pass  # pylint: disable=W0107

def show_main_topics_callback(main_topics : MainTopics): # pylint: disable=W0613
    """Show main topic information"""
    pass  # pylint: disable=W0107

def show_topics_score_callback(result_list : list[TopicScoreItem]): # pylint: disable=W0613
    """Show topic score"""
    pass  # pylint: disable=W0107

def run():
    """Main run cycle"""

    LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    if not LLM_OPENAI_API_KEY:
        logger.error('GPT KEY not found.')
        return

    site_map_only = False
    skip_translation = True
    override_by_url_words = False
    override_by_url_words_less = 0
    url_words_add = 0.2
    skip_summary = False
    use_topic_priority = True
    use_leaders = True

    backend_params = BackendParams(
        site_map_only,
        skip_translation,
        override_by_url_words,
        override_by_url_words_less,
        url_words_add,
        skip_summary,
        use_topic_priority,
        use_leaders,
        LLM_OPENAI_API_KEY,
        BackendCallbacks(
            report_status,
            report_substatus,
            used_tokens_callback,
            show_original_text_callback,
            report_error_callback,
            show_summary_callback,
            show_lang_callback,
            show_extracted_text_callback,
            show_debug_json_callback,
            show_main_topics_callback,
            show_topics_score_callback
        )
    )

    logger.debug('Create back-end instance...')
    back_end = BackEndCore(backend_params)

    input_url_list : list[str] = ['https://www.pmi.com/faq-section/smoking-and-cigarettes']

    read_mode = ReadModeHTML.BS4.value
    only_read_html = False

    logger.info('Run...')
    bulk_result : list[ScoreResultItem] = back_end.run(input_url_list, read_mode, only_read_html)

    logger.info('Resuls:')
    logger.debug(bulk_result)
    logger.info(f'total_used_tokens={total_used_tokens}')

if __name__ == '__main__':
    total_used_tokens = 0
    logger = init_root_logger()

    logger.info('Console app run...')
    run()