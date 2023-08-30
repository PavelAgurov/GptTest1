"""
    Main PMI
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121

import os
import pandas as pd

import streamlit as st

import strings
from utils_streamlit import streamlit_hack_remove_top_space
from backend.backend_core import BackEndCore, BackendParams, BackendCallbacks
from backend.base_classes import MainTopics, ScoreResultItem
from backend.bulk_output import BulkOutputParams

# https://discuss.streamlit.io/t/watching-custom-folders/1507/4
os.environ['PYTHONPATH'] = ';'.join([r"backend", r"backend\\llm"])

COST_1K = 0.002
OUTPUT_DATA_FILE = "result.csv"
FOOTER_LIST = ['Quick links']

# --------------------------------- Sessions

SESSION_TOKEN_COUNT = 'token_count'
if SESSION_TOKEN_COUNT not in st.session_state:
    st.session_state[SESSION_TOKEN_COUNT] = 0

# ------------------------------- UI

MODE_ONE   = 'One URL'
MODE_BULK  = 'Bulk mode'
MODE_EXCEL = 'Load from excel'

st.set_page_config(page_title="PMI Topics Demo", layout="wide")
st.title('PMI Topics Demo')
streamlit_hack_remove_top_space()

tab_process, tab_settings, tab_debug = st.tabs(["Process URL(s)", "Settings", "Debug"])

with tab_process:
    mode_selector              = st.radio(label="Mode", options=[MODE_ONE, MODE_BULK, MODE_EXCEL], index=0, horizontal=True, label_visibility="hidden")
    col_s1, col_s2, col_s3 = st.columns(3)
    inc_source_checbox         = col_s1.checkbox(label= "Include source in bulk output", disabled= mode_selector == MODE_ONE)
    inc_explanation_checkbox   = col_s2.checkbox(label= "Include explanation in bulk output", disabled= mode_selector == MODE_ONE)
    score_by_summary_checkbox  = col_s3.checkbox(label= "Score by summary", value=True)

    if mode_selector == MODE_ONE:
        input_url_one = st.text_input("URL: ", "", key="input")
    elif mode_selector == MODE_BULK:
        input_url_bulk = st.text_area("URLs: ", "", key="input")
    else:
        input_url_excel = st.file_uploader(
            'Excel with URLs',
            type=["xlsx"],
            accept_multiple_files= False,
            key="input_url_file"
        )
        excel_data_status = st.empty()

    _, col_button = st.columns([10, 1])
    run_button = col_button.button(label="RUN")
    st.divider()

    status_container  = st.empty()
    substatus_container  = st.empty()
    if mode_selector == MODE_ONE:
        org_text_container = st.expander(label="Original Text").empty()
        if score_by_summary_checkbox:
            summary_container  = st.expander(label="Summary").empty()
        lang_container     = st.empty()
        extracted_text_container = st.expander(label="Extracted (and translated) Text").empty()
    main_topics_container = st.container().empty()
    output_container = st.container().empty()
    if mode_selector != MODE_ONE:
        export_container = st.empty()
    debug_container = st.container()

with tab_settings:
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    footer_texts = st.text_area("Footers", value= '\n'.join(FOOTER_LIST))

with tab_debug:
    if mode_selector == MODE_ONE:
        debug_json_container = st.expander(label="Score json").empty()

with st.sidebar:
    header_container = st.container()
    header_container.markdown(strings.HOW_IT_WORK, unsafe_allow_html=True)
    token_container = st.empty()
    error_container = st.container()

def skip_callback():
    """Skip callback"""

def show_total_tokens():
    """Show total count of tokens"""
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
    if mode_selector == MODE_ONE and score_by_summary_checkbox:
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

def show_topics_score_callback(result_list : list): # TODO - replace list to the class
    """Show topic score"""
    if mode_selector != MODE_ONE:
        return
    df = pd.DataFrame(result_list, columns = ['Topic', 'Score', 'Explanation'])
    df = df.sort_values(by=['Score'], ascending=False)
    output_container.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_data
def convert_df_to_csv(df_csv : pd.DataFrame):
    """Convert DataFrame into csv and cahce it"""
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df_csv.to_csv().encode('utf-8')

if open_api_key:
    LLM_OPENAI_API_KEY = open_api_key
else:
    LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

show_total_tokens()

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
else:
    if not input_url_excel:
        report_status('Excel file was not uploaded')
        st.stop()
    excel_data = pd.read_excel(input_url_excel)
    input_url_list = excel_data.iloc[:,0].values
    excel_data_status.markdown(f'Loaded {len(input_url_list)} URLs')

backend_params = BackendParams(
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
    score_by_summary_checkbox,
    footer_texts.split('\n')
)

back_end = BackEndCore(backend_params)
bulk_result : list[ScoreResultItem] = back_end.run(input_url_list)

if mode_selector == MODE_ONE or bulk_result is None:
    st.stop()

# bulk processing
bulk_output_params = BulkOutputParams(
    inc_explanation_checkbox,
    inc_source_checbox
)
df_bulk = back_end.build_ouput_data(bulk_result, bulk_output_params)
output_container.dataframe(df_bulk, use_container_width=True, hide_index=True)

csv_data = convert_df_to_csv(df_bulk)
export_container.download_button(
    label='Download Excel', 
    data = csv_data,  
    file_name= OUTPUT_DATA_FILE, 
    mime='text/csv', 
    on_click= skip_callback
)
      
