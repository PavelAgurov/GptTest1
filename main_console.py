"""
    Console app
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,W1203,R0914

import os
import sys
import logging
import argparse
import toml

from backend.backend_core import BackEndCore, BackendParams, BackendCallbacks, ReadModeHTML
from backend.base_classes import ScoreResultItem, MainTopics, TopicScoreItem
from utils.app_logger import init_root_logger
from backend.bulk_output import BulkOutputParams

OUTPUT_EXTENSION = '.xlsx'

DEFAULT_CONFIG_FILE = r'.streamlit\secrets.toml'

total_used_tokens = 0
logger : logging.Logger

def report_status(status_str : str):
    """Show first status line"""
    if status_str:
        logger.info(status_str)
        with open('main_console.log', 'at', encoding="utf-8") as f:
            f.write(status_str + '\n')

def report_substatus(substatus_str : str):
    """Show second line of status"""
    if substatus_str:
        logger.debug(substatus_str)

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

def run(input_file : str, output_folder : str, all_secrets : dict[str, any], env_openai_key : str):
    """Main run cycle"""

    logger.info(f'Read URLs from file {input_file}')
    with open(input_file, "rt", encoding="utf-8") as f_input:
        input_url_list = f_input.readlines()

    if input_url_list:
        input_url_list = [u.strip() for u in input_url_list if not u.startswith('#')]

    if len(input_url_list) == 0:
        logger.error(f'Input file {input_file} has no URLs.')
        return

    logger.debug(f'Read {len(input_url_list)} URLs from file {input_file}')

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
        all_secrets,
        env_openai_key,
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

    read_mode = ReadModeHTML.BS4.value
    only_read_html = False

    logger.info('Run...')
    bulk_result : list[ScoreResultItem] = back_end.run(input_url_list, read_mode, only_read_html)

    logger.info('Resuls:')
    logger.debug(bulk_result)
    logger.info(f'total_used_tokens={total_used_tokens}')

    bulk_output_params = BulkOutputParams(
        True,
        True,
        False,
        False
    )

    df_bulk_result = back_end.build_ouput_data(bulk_result, bulk_output_params)
    if df_bulk_result.error:
        logger.error(df_bulk_result.error)
        sys.exit()

    if df_bulk_result.data is None:
        logger.error(f'Result data for package {input_file} is empty')
        sys.exit()

    output_file = f'{input_file}{OUTPUT_EXTENSION}'
    if output_folder:
        output_file = os.path.join(output_folder, os.path.basename(output_file))

    if not output_file.endswith(OUTPUT_EXTENSION):
        output_file = f'{output_file}{OUTPUT_EXTENSION}'

    try:
        logger.info(f'Saving output {output_file}')
        df_bulk_result.data.to_excel(output_file, index= False)
        logger.debug(f'Output {output_file} saved')
    except Exception as saving_error: # pylint: disable=W0718
        logger.error(saving_error)

def main():
    """Main procedure"""

    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Input", help = "Input file name with URLs (txt)", required=False)
    parser.add_argument("-p", "--Package", help = "Package file name (txt)", required=False)
    parser.add_argument("-o", "--Output", help = "Output folder name", required=False)
    parser.add_argument("-c", "--Config", help = "Config file (toml)", required=False)
    args = parser.parse_args()

    if not args.Input and not args.Package:
        parser.error('--Input or --Package argument is required')
    if args.Input and args.Package:
        parser.error('Only one --Input or --Package argument is allowed')

    input_file_list = []
    if args.Input:
        input_file_list = [args.Input]
        logger.info(f'Input file: {args.Input}.')

    if args.Package:
        logger.info(f'Read package list from file {args.Package}')
        with open(args.Package, "rt", encoding="utf-8") as f:
            input_file_list = f.readlines()
        logger.info(f'Read {len(input_file_list)} input files from {args.Package}')

    if input_file_list:
        input_file_list = [f.strip() for f in input_file_list if len(f.strip()) > 0]

    if not input_file_list:
        logger.error('No input files provided')
        sys.exit()

    output_folder = args.Output
    if output_folder:
        os.makedirs(output_folder, exist_ok= True)
    logger.info(f'Output folder: {args.Output}')

    env_openai_key = None
    config = None
    settings_config_file = DEFAULT_CONFIG_FILE
    if args.Config: # if config is provided - file must exist
        settings_config_file = args.Config
        if not os.path.isfile(settings_config_file):
            logger.error(f'Config file {settings_config_file} not found.')
            sys.exit()

    # if we have default config file or provided config - use it
    if os.path.isfile(settings_config_file):
        with open(settings_config_file, 'r', encoding="utf-8") as f:
            config = toml.load(f)
    else:
        # if there is no config file - we should have env variable
        env_openai_key = os.environ.get("OPENAI_API_KEY")
        if not env_openai_key:
            logger.error('GPT KEY not found.')
            sys.exit()


    logger.info('Start processing...')
    for index, input_file in enumerate(input_file_list):
        if not input_file:
            continue
        if input_file.startswith('#'):
            continue

        progress_str = f'Process {input_file} ({index+1}/{len(input_file_list)})'
        logger.info(progress_str)

        try:

            # try in the same folder as package file
            if not os.path.isfile(input_file) and args.Package:
                if not os.path.dirname(input_file):
                    input_file = os.path.join(os.path.dirname(args.Package), input_file)

            run(input_file, output_folder, config, env_openai_key)
        except Exception as run_error: # pylint: disable=W0718
            logger.error(run_error)

if __name__ == '__main__':
    total_used_tokens = 0
    logger = init_root_logger()
    main()