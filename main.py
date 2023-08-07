import pandas as pd
import json
import os
import streamlit as st
import langchain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
import traceback
from langchain.callbacks import get_openai_callback
import tiktoken
import collections
from refine import RefineChain
from topics import TOPICS_LIST
import utils
import prompts
import strings


MODEL_NAME = "gpt-3.5-turbo"
COST_1K = 0.002
MAX_TOKENS_TRANSLATION = 1000
MAX_TOKENS_SCORE = 2000
MAX_TOKENS_SUMMARY = 2500
FIRST_PARAGRAPH_MAX_TOKEN = 200 # small text to check language

OUTPUT_DATA_FILE = "result.csv"

FOOTER_LIST = ['Quick links']

st.set_page_config(page_title="PMI Topics Demo", page_icon=":robot:")
st.title('PMI Topics Demo')

tab_process, tab_settings, tab_debug = st.tabs(["Process URL(s)", "Settings", "Debug"])

with tab_process:
    header_container = st.container()
    bulk_mode_checkbox         = st.checkbox(label= "Bulk mode")
    inc_source_checbox         = st.checkbox(label= "Include source in bulk output", disabled=not bulk_mode_checkbox)
    inc_explanation_checkbox   = st.checkbox(label= "Include explanation in bulk output", disabled=not bulk_mode_checkbox)
    score_by_summary_checkbox  = st.checkbox(label= "Score by summary")
    if not bulk_mode_checkbox:
        input_url = st.text_input("URL: ", "", key="input")
    else:
        input_url = st.text_area("URLs: ", "", key="input")
    status_container  = st.empty()
    if not bulk_mode_checkbox:
        org_text_container = st.expander(label="Original Text").empty()
        if score_by_summary_checkbox:
            summary_container  = st.expander(label="Summary").empty()
        lang_container     = st.empty()
        text_container     = st.expander(label="Extracted (and translated) Text").empty()
    main_topics_container = st.container().empty()
    output_container = st.container().empty()
    if bulk_mode_checkbox:
        export_container = st.empty()
    debug_container = st.container()

with tab_settings:
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    footer_texts = st.text_area("Footers", value= '\n'.join(FOOTER_LIST))

with tab_debug:
    if not bulk_mode_checkbox:
        debug_json_container = st.container()

with st.sidebar:
    token_container = st.empty()
    error_container = st.container()

header_container.markdown(strings.how_it_work, unsafe_allow_html=True)

from unstructured.partition.html import partition_html

def skip_callback():
    pass

def load_html(url):
  elements = partition_html(url=url)
  return "\n\n".join([str(el) for el in elements])

def grouper(iterable, step):
    result = []
    for i in range(0, len(iterable), step):
        result.append(iterable[i:i+step])
    return result

def num_tokens_from_string(string, llm_encoding):
    return len(llm_encoding.encode(string))

def sort_dict_by_value(d, reverse = False):
  return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

def show_total_tokens(n):
     token_container.markdown(f'Tokens used: {n} (~cost ${n/1000*COST_1K:10.4f})')

def text_extractor(text):
    footer_text_list = footer_texts.split('\n')
    for f in footer_text_list:
        f = f.strip()
        if len(f) == 0:
            continue
        footer_index = text.find(f)
        if footer_index != -1:
            text = text[:footer_index]
    return text

def text_to_paragraph(extracted_text, token_estimator) -> []:
    result_paragraph_list = []
    extracted_sentence_list = extracted_text.split('\n')

    current_token_count = 0
    current_paragraph   = []
    for i, p in enumerate(extracted_sentence_list):
        max_tokens = FIRST_PARAGRAPH_MAX_TOKEN
        if len(result_paragraph_list) > 0: # first paragpath found
            max_tokens = MAX_TOKENS_TRANSLATION
        token_count_p = len(token_estimator.encode(p))
        if ((current_token_count + token_count_p) < max_tokens):
            current_paragraph.append(p)
            current_token_count = current_token_count + token_count_p
        else:
            result_paragraph_list.append('\n\n'.join(current_paragraph))
            current_paragraph  = [p]
            current_token_count = token_count_p
    if len(current_paragraph) > 0:
        result_paragraph_list.append('\n\n'.join(current_paragraph))
    return result_paragraph_list

@st.cache_data
def convert_df_to_csv(df : pd.DataFrame):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

if open_api_key:
    LLM_OPENAI_API_KEY = open_api_key
else:
    LLM_OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

langchain.llm_cache = SQLiteCache()

llm_translation = ChatOpenAI(model_name = MODEL_NAME, openai_api_key = LLM_OPENAI_API_KEY, max_tokens = MAX_TOKENS_TRANSLATION)
translation_prompt= PromptTemplate.from_template(prompts.translation_prompt_template)
translation_chain  = LLMChain(llm=llm_translation, prompt = translation_prompt)

llm_score = ChatOpenAI(model_name = MODEL_NAME, openai_api_key = LLM_OPENAI_API_KEY, max_tokens = MAX_TOKENS_SCORE)
score_prompt = PromptTemplate.from_template(prompts.score_prompt_template)
score_chain  = LLMChain(llm=llm_score, prompt = score_prompt)

llm_summary = ChatOpenAI(model_name = MODEL_NAME, openai_api_key = LLM_OPENAI_API_KEY, max_tokens = MAX_TOKENS_SUMMARY)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name= MODEL_NAME, model_name=MODEL_NAME, chunk_size=1000, chunk_overlap=0)

token_estimator = tiktoken.encoding_for_model("gpt-3.5-turbo")

TOPIC_CHUNKS = [TOPICS_LIST] # grouper(TOPICS_LIST, 4)
TOPIC_DICT   =  {t[0]:t[1] for t in TOPICS_LIST}

total_token_count = 0
show_total_tokens(total_token_count)

if not input_url:
    st.stop()

input_url_list = input_url.split('\n')

bulk_result_list = []
for index_url, current_url in enumerate(input_url_list):
    current_url = current_url.strip()
    if not current_url:
        continue

    url_index_str = ''
    if bulk_mode_checkbox:
        url_index_str = f'[url: {index_url+1}/{len(input_url_list)}]'

    status_container.markdown(f'Request URL {url_index_str} "{current_url}"...')
    input_text = load_html(current_url)
    input_text_len = len(input_text)
    status_container.markdown(f'Done. Got {input_text_len} chars.')
    input_text = input_text.replace("“", "'").replace("“", "”").replace("\"", "'")
    if not bulk_mode_checkbox:
        org_text_container.markdown(input_text)

    if score_by_summary_checkbox:
        status_container.markdown(f'Split documents for refining...')
        split_docs = text_splitter.split_documents([Document(page_content = input_text)])
        status_container.markdown(f'')

        status_container.markdown(f'Request LLM for summary {url_index_str}...')
        refine_result = RefineChain(llm_summary).refine(split_docs)
        summary = ""
        if not refine_result.error:
            summary = refine_result.summary
        total_token_count += refine_result.tokens_used
        show_total_tokens(total_token_count)
        if not bulk_mode_checkbox:
            summary_container.markdown(summary)
        extracted_text = text_extractor(summary)
        status_container.markdown(f'Summary is ready')
    else:
        extracted_text = text_extractor(input_text)
    extracted_text_len = len(extracted_text)

    if not score_by_summary_checkbox:
        paragraph_list = text_to_paragraph(extracted_text, token_estimator)
    else:
        paragraph_list = [extracted_text]
    
    translated_list = []
    translated_lang = "None"
    no_translation = False
    for i, p in enumerate(paragraph_list):
        status_container.markdown(f'Request LLM for translation {url_index_str} paragraph: {i+1}/{len(paragraph_list)}...')
        with get_openai_callback() as cb:
            translated_text = translation_chain.run(article = p)
        total_token_count += cb.total_tokens
        show_total_tokens(total_token_count)
        try:
            translated_text_json = json.loads(utils.get_fixed_json(translated_text))
            translated_lang = translated_text_json["lang"]
            if translated_lang == "English" or translated_lang == "en":
                no_translation = True
                status_container.markdown(f'Text is in English. No translation needed.')
                break
            translated_text = translated_text_json["translated"]
            translated_list.append(translated_text)
        except:
            no_translation = True

    if not no_translation:
        if not bulk_mode_checkbox:
            lang_container.markdown(f'Language of original text: {translated_lang}')
    else:
        if not bulk_mode_checkbox:
            lang_container.markdown("Text is in English. No translation needed.")
        translated_list = paragraph_list # just use "as is"
    
    full_translated_text = '\n'.join(translated_list)
    transpated_text_len = len(full_translated_text)
    if not bulk_mode_checkbox:
        text_container.markdown(full_translated_text)

    result_score = {}
    result_primary_topic_json   = None
    result_secondary_topic_json = None

    for i, p in enumerate(translated_list):
        for j, topic_def in enumerate(TOPIC_CHUNKS):
            topics_for_prompt = "\n".join([f'{t[0]}. {t[2]}' if len(t[2])>0 else f'{t[0]}. {t[1]}' for t in topic_def])
            topics_id_name_list = {t[0]:t[1] for t in topic_def}

            status_container.markdown(f'Request LLM to score {url_index_str} paragraph: {i+1}/{len(translated_list)}, topics chunk: {j+1}/{len(TOPIC_CHUNKS)}...')
            with get_openai_callback() as cb:
                extracted_score = score_chain.run(topics = topics_for_prompt, article = p, url = current_url)
            total_token_count += cb.total_tokens
            show_total_tokens(total_token_count)
            status_container.markdown(f'Done. Got {len(extracted_score)} chars.')
            if not bulk_mode_checkbox:
                debug_json_container.markdown(extracted_score)

            try:
                status_container.markdown('Extract result...')
                extracted_score_json = json.loads(utils.get_fixed_json(extracted_score))
                status_container.markdown(f'')

                primary_topic_json = extracted_score_json['primary_topic']
                if not result_primary_topic_json or result_primary_topic_json['score'] < primary_topic_json['score']:
                    result_primary_topic_json = primary_topic_json

                secondary_topic_json = extracted_score_json['secondary_topic']
                if not result_secondary_topic_json or result_secondary_topic_json['score'] < secondary_topic_json['score']:
                    result_secondary_topic_json = secondary_topic_json

                for t in extracted_score_json['topics']:
                    current_score = 0
                    topic_id = t["topicID"]
                    if topic_id in result_score:
                        current_score = result_score[topic_id][0]
                    new_score = t["score"]
                    if (new_score > current_score) or (current_score == 0):
                        result_score[topic_id] = [new_score, t["explanation"]]

            except Exception as error:
                output_container.markdown(f'Error:\n\n{extracted_score}\n\nError: {error}\n\n{traceback.format_exc()}', unsafe_allow_html=True)
                st.stop()
        
    show_total_tokens(total_token_count)
    
    main_topics_result = []
    primary_topic = ""
    primary_topic_score = 0
    primary_topic_explanation =""
    if result_primary_topic_json:
        primary_topic_score = result_primary_topic_json['score']
        primary_topic_explanation = result_primary_topic_json['explanation']
        primary_topic = TOPIC_DICT[result_primary_topic_json['topic_id']]
        main_topics_result.append(['Primary', primary_topic, primary_topic_score, primary_topic_explanation])

    secondary_topic = ""
    secondary_topic_score = 0
    secondary_topic_explanation = ""
    if result_secondary_topic_json:
        secondary_topic_score = result_secondary_topic_json['score']
        secondary_topic_explanation = result_secondary_topic_json['explanation']
        secondary_topic = TOPIC_DICT[result_secondary_topic_json['topic_id']]
        main_topics_result.append(['Secondary', secondary_topic, secondary_topic_score, secondary_topic_explanation])

    if not bulk_mode_checkbox:
        df = pd.DataFrame(main_topics_result, columns = ['#', 'Topic', 'Score', 'Explanation'])
        main_topics_container.dataframe(df, use_container_width=True, hide_index=True)

    ordered_result_score = collections.OrderedDict(sorted(result_score.items()))
    result_list = []
    for s in result_score.keys():
        if s in TOPIC_DICT:
            result_list.append([TOPIC_DICT[s], *ordered_result_score[s]])
        else:
            error_container.markdown(ordered_result_score)

    bulk_row = [
                current_url, input_text_len, extracted_text_len, translated_lang, transpated_text_len, 
                primary_topic, primary_topic_score, primary_topic_explanation,
                secondary_topic, secondary_topic_score, secondary_topic_explanation,
                full_translated_text, # 11 - source text
                ordered_result_score  # 12 - topic data
               ]
    bulk_result_list.append(bulk_row)

    if not bulk_mode_checkbox:
        df = pd.DataFrame(result_list, columns = ['Topic', 'Score', 'Explanation'])
        df = df.sort_values(by=['Score'], ascending=False)
        output_container.dataframe(df, use_container_width=True, hide_index=True)

if bulk_mode_checkbox:
    bulk_columns = ['URL', 'Input length', 'Extracted length', 'Lang', 'Translated length']
    bulk_columns.extend(['Primary', 'Primary score'])
    if inc_explanation_checkbox:
        bulk_columns.extend(['Primary explanation'])
    bulk_columns.extend(['Secondary', 'Secondary score'])
    if inc_explanation_checkbox:
        bulk_columns.extend(['Secondary explanation'])
    if inc_source_checbox:
        bulk_columns.extend(['Source text'])
    for t in TOPICS_LIST:
        bulk_columns.extend([f'[{t[0]}]{t[1]}'])
        if inc_explanation_checkbox:
            bulk_columns.extend([f'[{t[0]}]Explanation'])
    bulk_data = []
    for row in bulk_result_list:
        bulk_row = []
        bulk_row.extend([row[0], row[1], row[2], row[3], row[4]])

        bulk_row.extend([row[5], row[6]]) # primary topic
        if inc_explanation_checkbox:
            bulk_row.extend([row[7]])

        bulk_row.extend([row[8], row[9]]) # secondary topic
        if inc_explanation_checkbox:
            bulk_row.extend([row[10]])

        if inc_source_checbox:
            bulk_row.extend([row[11]])
        
        score_data = row[12]
        for t in TOPICS_LIST: 
            if t[0] in score_data:
                topic_score = score_data[t[0]]
                bulk_row.extend([topic_score[0]])
                if inc_explanation_checkbox:
                    bulk_row.extend([topic_score[1]])
            else:
                bulk_row.extend([0])
                if inc_explanation_checkbox:
                    bulk_row.extend([''])

        bulk_data.append(bulk_row)
    df = pd.DataFrame(bulk_data, columns = bulk_columns)
    output_container.dataframe(df, use_container_width=True, hide_index=True)

    data = convert_df_to_csv(df)
    export_container.download_button(label='Download Excel', data = data,  file_name= OUTPUT_DATA_FILE, mime='text/csv', on_click= skip_callback)
      
