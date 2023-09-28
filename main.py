"""
    Main PMI
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121

import os
import time
import pandas as pd

import streamlit as st

import strings
from utils_streamlit import streamlit_hack_remove_top_space
from backend.backend_core import BackEndCore, BackendParams, BackendCallbacks, ReadModeHTML
from backend.base_classes import MainTopics, ScoreResultItem, TopicScoreItem
from backend.bulk_output import BulkOutputParams
from backend.topic_manager import TopicManager, TopicDefinition
from sitemap_utils import sitemap_load

# https://discuss.streamlit.io/t/watching-custom-folders/1507/4
os.environ['PYTHONPATH'] = ';'.join([r"backend", r"backend\\llm"])

COST_1K = 0.002
OUTPUT_DATA_FILE = "result.csv"
OUTPUT_TOPIC_FILE = "topics.csv"
FOOTER_LIST = ['Quick links']

# --------------------------------- Sessions

SESSION_TOKEN_COUNT = 'token_count'

# ------------------------------- UI

MODE_ONE   = 'One URL'
MODE_BULK  = 'Bulk mode'
MODE_EXCEL = 'Load from excel'
MODE_SITEMAP = 'From Sitemap'

EXCLUDED_PREFIXES_URLS = [
    "https://www.pmi.com/markets/egypt/ar",
    "https://www.pmi.com/protected-area"
]

EXCLUDED_URLS = [
    "https://www.pmi.com"
]

st.set_page_config(page_title="PMI Topics Demo", layout="wide")
st.title('PMI Topics Demo')
streamlit_hack_remove_top_space()

tab_process, tab_settings, tab_topic_editor, tab_debug = st.tabs(["Process URL(s)", "Settings", "Topics", "Debug"])

site_map_only = False
with tab_process:
    mode_selector              = st.radio(label="Mode", options=[MODE_ONE, MODE_BULK, MODE_EXCEL, MODE_SITEMAP], index=0, horizontal=True, label_visibility="hidden")
    col_s1, col_s2, col_s3 = st.columns(3)
    inc_source_checbox         = col_s1.checkbox(label= "Include source in bulk output", disabled= mode_selector == MODE_ONE)
    inc_explanation_checkbox   = col_s2.checkbox(label= "Include explanation in bulk output", disabled= mode_selector == MODE_ONE)
    add_gold_data_checkbox     = col_s3.checkbox(label= "Add golden data", value=True)

    read_mode_list = [e.value for e in ReadModeHTML]
    read_mode = st.radio(
            "HTML Read mode:",
            key="html_read_mode",
            options= read_mode_list,
            index=1,
            horizontal=True
    )

    only_read_html = st.checkbox(label='Only read HTML', value= False)

    if mode_selector == MODE_ONE:
        input_url_one = st.text_input("URL: ", "", key="input")
    elif mode_selector == MODE_BULK:
        input_url_bulk = st.text_area("URLs: ", "", key="input")
    elif mode_selector == MODE_EXCEL:
        input_url_excel = st.file_uploader(
            'Excel with URLs',
            type=["xlsx"],
            accept_multiple_files= False,
            key="input_url_file"
        )
        excel_data_status = st.empty()
    else:
        col_sm1, col_sm2, col_sm3 = st.columns(3)
        site_map_only  = st.checkbox('Only build sitemap')
        input_sitemap  = col_sm1.text_input("Sitemap URL: ", "", key="input")
        site_map_from  = col_sm2.number_input("From:", min_value=1, max_value= 10000, value=1)
        site_map_limit = col_sm3.number_input("Max count ('0' means 'no limit'):", min_value=0, max_value= 10000, value=100)
        sime_map_exluded_prefixes = st.text_area(label="Excluded URL prefixes:", value='\n'.join(EXCLUDED_PREFIXES_URLS))
        sime_map_exluded_urls = st.text_area(label="Excluded URLs:", value='\n'.join(EXCLUDED_URLS))
        sitemap_data_status = st.empty()

    _, col_button = st.columns([10, 1])
    run_button = col_button.button(label="RUN")
    st.divider()

    status_container  = st.empty()
    substatus_container  = st.empty()
    if mode_selector == MODE_ONE:
        org_text_container = st.expander(label="Original Text").empty()
        summary_container  = st.expander(label="Summary").empty()
        lang_container     = st.empty()
        extracted_text_container = st.expander(label="Extracted (and translated) Text").empty()
    main_topics_container = st.container().empty()
    output_container = st.container().empty()
    if mode_selector != MODE_ONE:
        export_container = st.empty()
    debug_container = st.container()
    bulk_error_container = st.empty()

with tab_settings:
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    skip_translation = st.checkbox(label= "Skip translation", value=True)
    priority_threshold_main = st.number_input(label="Priority threshold for primary/secondary (1 - always, 0 - never)", 
                                              min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    footer_texts = st.text_area("Footers", value= '\n'.join(FOOTER_LIST))

with tab_topic_editor:
    topic_editor_container = st.container()
    topic_editor_info = st.empty()
    topic_editor_control = None
    col_te_1, col_te_2 = st.columns([60, 10])
    export_topic_editor_container = col_te_1.empty()
    topics_reset_button = col_te_2.button(label="Reset all topics")

with tab_debug:
    if mode_selector == MODE_ONE:
        debug_json_container = st.expander(label="Score json").empty()

with st.sidebar:
    header_container = st.container()
    header_container.markdown(strings.HOW_IT_WORK, unsafe_allow_html=True)
    token_container = st.empty()
    start_time_container = st.empty()
    error_container = st.container()

def skip_callback():
    """Skip callback"""

def show_total_tokens():
    """Show total count of tokens"""
    if SESSION_TOKEN_COUNT not in st.session_state:
        st.session_state[SESSION_TOKEN_COUNT] = 0
    token_counter = st.session_state[SESSION_TOKEN_COUNT]
    token_container.markdown(f'Tokens used: {token_counter} (~cost ${token_counter/1000*COST_1K:10.4f})')

def report_status(status_str : str):
    """Show first status line"""
    status_container.markdown(status_str)

def report_substatus(substatus_str : str):
    """Show second line of status"""
    substatus_container.markdown(substatus_str)

def used_tokens_callback(used_tokens : int):
    """Update token counter"""
    if SESSION_TOKEN_COUNT not in st.session_state:
        st.session_state[SESSION_TOKEN_COUNT] = 0
    n = st.session_state[SESSION_TOKEN_COUNT]
    n += used_tokens
    st.session_state[SESSION_TOKEN_COUNT] = n
    show_total_tokens()

def show_original_text_callback(text : str):
    """Show original text"""
    if mode_selector == MODE_ONE:
        org_text_container.markdown(text)

def show_summary_callback(summary : str):
    """Show summary"""
    if mode_selector == MODE_ONE:
        summary_container.markdown(summary)

def report_error_callback(err : str):
    """Report error"""
    error_container.markdown(err)

def show_lang_callback(lang_str : str):
    """Show lang string"""
    if mode_selector == MODE_ONE:
        lang_container.markdown(lang_str)

def show_extracted_text_callback(text : str):
    """Show extracted text"""
    if mode_selector == MODE_ONE:
        extracted_text_container.markdown(text)

def show_debug_json_callback(json_str : str):
    """Show debug json"""
    if mode_selector == MODE_ONE:
        debug_json_container.markdown(json_str)

def show_main_topics_callback(main_topics : MainTopics):
    """Show main topic information"""
    if mode_selector != MODE_ONE:
        return
    main_topics_result = []
    main_topics_result.append([
        'Primary',
        main_topics.primary.topic,
        main_topics.primary.topic_score,
        main_topics.primary.explanation
    ])
    main_topics_result.append([
        'Secondary',
        main_topics.secondary.topic,
        main_topics.secondary.topic_score,
        main_topics.secondary.explanation
    ])

    df = pd.DataFrame(main_topics_result, columns = ['#', 'Topic', 'Score', 'Explanation'])
    main_topics_container.dataframe(df, use_container_width=True, hide_index=True)

def show_topics_score_callback(result_list : list[TopicScoreItem]):
    """Show topic score"""
    if mode_selector != MODE_ONE:
        return
    output_list = [[r.topic, r.topic_score, r.explanation ]for r in result_list]
    df = pd.DataFrame(output_list, columns = ['Topic', 'Score', 'Explanation'])
    df = df.sort_values(by=['Score'], ascending=False)
    output_container.dataframe(df, use_container_width=True, hide_index=True)

def update_topic_editor(updated_editor : pd.DataFrame, topic_manager: TopicManager) -> bool:
    """Changed data in topic editor"""
    saved_editor = pd.DataFrame([[t.id, t.name, t.description] for t in topic_manager.get_topic_list()], columns=["Id", "Name", "Description"])
    if saved_editor is not None and updated_editor.equals(saved_editor):
        return False
    updated = [TopicDefinition(*row) for row in updated_editor.values.tolist()]
    topic_manager.save_topic_descriptions(updated)
    return True

def show_topic_editor(topic_manager : TopicManager):
    """Show topic editor"""
    data = pd.DataFrame([[t.id, t.name, t.description] for t in topic_manager.get_topic_list()], columns=["Id", "Name", "Description"])
    editor_control = topic_editor_container.data_editor(
        data,
        disabled= ["Id", "Name"],
        use_container_width=True,
        hide_index=True
    )
    return editor_control

@st.cache_data
def convert_df_to_csv(df_csv : pd.DataFrame):
    """Convert DataFrame into csv and cahce it"""
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df_csv.to_csv().encode('utf-8')

if open_api_key:
    LLM_OPENAI_API_KEY = open_api_key
else:
    LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

backend_params = BackendParams(
    site_map_only,
    skip_translation,
    priority_threshold_main,
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
    ),
    footer_texts.split('\n')
)

back_end = BackEndCore(backend_params)

topic_editor_control = show_topic_editor(back_end.topic_manager)
if update_topic_editor(topic_editor_control, back_end.topic_manager):
    st.experimental_rerun()

show_total_tokens()

if topics_reset_button:
    back_end.topic_manager.reset_all_topics()
    st.experimental_rerun()

csv_topic_data = convert_df_to_csv(topic_editor_control)
export_topic_editor_container.download_button(
    label='Download Topic Data', 
    data = csv_topic_data,  
    file_name= OUTPUT_TOPIC_FILE, 
    mime='text/csv', 
    on_click= skip_callback
)

if not run_button:
    st.stop()

input_url_list = []
if mode_selector == MODE_ONE:
    if not input_url_one:
        report_status('URL was not provided')
        st.stop()
    input_url_list = [input_url_one]
elif mode_selector == MODE_BULK:
    if not input_url_bulk:
        report_status('URL(s) were not provided')
        st.stop()
    input_url_list = input_url_bulk.split('\n')
elif mode_selector == MODE_EXCEL:
    if not input_url_excel:
        report_status('Excel file was not uploaded')
        st.stop()
    excel_data = pd.read_excel(input_url_excel)
    input_url_list = excel_data.iloc[:,0].values
    excel_data_status.markdown(f'Loaded {len(input_url_list)} URLs')
else:
    if not input_sitemap:
        report_status('Sitemap URL was not provided')
        st.stop()
    sime_map_exluded_prefix_list = sime_map_exluded_prefixes.split()
    sime_map_exluded_urls_list = sime_map_exluded_urls.split()
    sitemap_result = sitemap_load(input_sitemap, sime_map_exluded_prefix_list, sime_map_exluded_urls_list)
    if sitemap_result.error:
        sitemap_data_status.markdown(f'ERROR: {sitemap_result.error}')
        st.stop()
    input_url_list = sitemap_result.url_list
    site_map_total_count = len(input_url_list)
    input_url_list = input_url_list[site_map_from-1:] # apply min
    if site_map_limit>0:
        input_url_list = input_url_list[:site_map_limit] # apply max
    sitemap_data_status.markdown(f'Loaded {len(input_url_list)} URLs (total count: {site_map_total_count})')

input_url_list = [u for u in input_url_list if len(u)> 0 and not u.startswith('#')]

start_time = time.localtime()
start_time_str = time.strftime("%H:%M:%S", start_time)
start_time_container.markdown(f'Start {start_time_str}')

bulk_result : list[ScoreResultItem] = back_end.run(input_url_list, read_mode, only_read_html)

end_time = time.localtime()
end_time_str = time.strftime("%H:%M:%S", start_time)
duration = (time.mktime(end_time) - time.mktime(start_time)) / 60
start_time_container.markdown(f'Start: {start_time_str}. End: {end_time_str}. Duration: {duration:.2f} sec.')

if mode_selector == MODE_ONE or bulk_result is None:
    st.stop()

# bulk processing
bulk_output_params = BulkOutputParams(
    inc_explanation_checkbox,
    inc_source_checbox,
    add_gold_data_checkbox,
    site_map_only
)
df_bulk_result = back_end.build_ouput_data(bulk_result, bulk_output_params)
if df_bulk_result.error:
    bulk_error_container.markdown(df_bulk_result.error)
output_container.dataframe(df_bulk_result.data, use_container_width=True, hide_index=True)

csv_data = convert_df_to_csv(df_bulk_result.data)
export_container.download_button(
    label='Download Excel', 
    data = csv_data,  
    file_name= OUTPUT_DATA_FILE, 
    mime='text/csv', 
    on_click= skip_callback
)
      
