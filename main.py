import pandas as pd
import json
import os
import streamlit as st
import langchain
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
import traceback
from langchain.callbacks import get_openai_callback
import tiktoken
import collections

MODEL_NAME = "gpt-3.5-turbo"
COST_1K = 0.002
MAX_TOKENS_TRANSLATION = 1000
MAX_TOKENS_SCORE = 2000
FIRST_PARAGRAPH_MAX_TOKEN = 200 # small text to check language

OUTPUT_DATA_FILE = "result.csv"

FOOTER_LIST = ['Quick links']

how_it_work_one = """\
First please put your openAPI key into settings.
Then insert one or list of URLs and check output.
"""

translation_prompt_template = """/
You are the best English translator. Please translate provided article (delimited with XML tags) into English.
Please provide result in JSON format with fields:
- lang (human language of original article - English, Russian, German etc.)
- translated (translated article)
Be sure that result is real JSON.

<article>{article}</article>
"""

score_prompt_template = """/
You are text classification machine. 
You have numerated list of topics:
{topics}

You task is to check if each topic is relevant to the provided article (delimited with XML tags) and explain why.
Also take into considiration article's URL (delimited with XML tags) that can be also relevant or not.
URL is very important information that can bring high score for the relevant topic.

Think about it step by step. 

Also add score of relevance from 0 to 1 (0 - not relevant, 1 - fully relevant).
Note that no two topics can have the same score, all scores must be different.
When possible you should include parts of original text to make explanation more clear.
Provide as much arguments as possible why article is related or not to the topic.
Be very scrupulous when you do classification. If it's only one or two words then it's not enough to be relevant.
When article can be considered as related to the topic, but does not provide any information - reduce score.
Validate all provided topics one by one.

Provide your output in json format with the keys: TopicID, Topic, Score, Explanation.

Example output:
[
{{"TopicID": 1, "Topic": "Tobacco Harm reduction", "Score": 0.5, "Explanation": "why article or URL are relevant or not"}},
{{"TopicID": 2, "Topic": "Tobaco science", "Score": 0, "Explanation": "some text here"}},
]

<article>{article}</article>
<article_url>{url}</article_url>
"""

TOPICS_LIST = [
  [1, "Tobacco Harm reduction", ""], 
  [2, "Tobacco multi-product approach",  ""],
  [3, "Inclusion, Diversity",  ""],
  [4, "Leadership content", "Leadership, strategies, interviews, communication, team management"],
  [5, "Investor Relations",  ""],
  [6, "Tobacco science",  ""],
  [7, "Smoke-free vision",  ""],
  [8, "Wellness, Healthcare, Health", ""]
]

st.set_page_config(page_title="PMI Topics Demo", page_icon=":robot:")
st.title('PMI Topics Demo')

tab_one, tab_settings = st.tabs(["Process one URL", "Settings"])

with tab_one:
    header_container = st.container()
    bulk_mode_checkbox         = st.checkbox(label= "Bulk mode")
    inc_source_checbox         = st.checkbox(label= "Include source in bulk output", disabled=not bulk_mode_checkbox)
    inc_explanation_checkbox   = st.checkbox(label= "Include explanation in bulk output", disabled=not bulk_mode_checkbox)
    score_by_summary_checkbox  = st.checkbox(label= "Score by summary")
    add_url_score_checkbox     = st.checkbox(label= "Add URL score")
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
    output_container = st.container().empty()
    if bulk_mode_checkbox:
        export_container = st.empty()
    debug_container = st.container()

with tab_settings:
    open_api_key = st.text_input("OpenAPI Key: ", "", key="open_api_key")
    footer_texts = st.text_area("Footers", value= '\n'.join(FOOTER_LIST))

with st.sidebar:
    token_container = st.empty()
    error_container = st.container()

header_container.markdown(how_it_work_one, unsafe_allow_html=True)

from unstructured.partition.html import partition_html

def skip_callback():
    pass

def load_html(url):
  elements = partition_html(url=url)
  return "\n\n".join([str(el) for el in elements])

def get_json(text):
    text = text.replace(", ]", "]").replace(",]", "]").replace(",\n]", "]")
    open_bracket = min(text.find('['), text.find('{'))
    if open_bracket == -1:
        return text
            
    close_bracket = max(text.rfind(']'), text.rfind('}'))
    if close_bracket == -1:
        return text
    return text[open_bracket:close_bracket+1]

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

def text_to_paragraph(extracted_text, token_estimator):
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
translation_prompt= PromptTemplate.from_template(translation_prompt_template)
translation_chain  = LLMChain(llm=llm_translation, prompt = translation_prompt)

llm_score = ChatOpenAI(model_name = MODEL_NAME, openai_api_key = LLM_OPENAI_API_KEY, max_tokens = MAX_TOKENS_SCORE)
score_prompt = PromptTemplate.from_template(score_prompt_template)
score_chain  = LLMChain(llm=llm_score, prompt = score_prompt)

llm_summary = ChatOpenAI(model_name = MODEL_NAME, openai_api_key = LLM_OPENAI_API_KEY, max_tokens = MAX_TOKENS_SCORE)
model_summary = load_summarize_chain(llm=llm_summary, chain_type = "refine")

token_estimator = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Long English: https://www.pmi.com/us/about-us/our-leadership-team
# Non English https://www.pmi.com/markets/portugal/pt/news/details?articleId=tabaqueira-participa-na-consulta-pública-para-a-estratégia-nacional-de-luta-contra-o-cancro-2021-2030

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
    if not bulk_mode_checkbox:
        org_text_container.markdown(input_text)

    if score_by_summary_checkbox:
        status_container.markdown(f'Request LLM for summary {url_index_str}...')
        with get_openai_callback() as cb:
            summary = model_summary.run([Document(page_content = input_text)])
            total_token_count += cb.total_tokens
            show_total_tokens(total_token_count)
        if not bulk_mode_checkbox:
            summary_container.markdown(summary)
        extracted_text = text_extractor(summary)
        status_container.markdown(f'Summary is ready')
    else:
        extracted_text = text_extractor(input_text)
    extracted_text_len = len(extracted_text)

    paragraph_list = text_to_paragraph(extracted_text, token_estimator)

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
            translated_text_json = json.loads(get_json(translated_text))
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
    transpated_text_len = len('\n'.join(translated_list))
    if not bulk_mode_checkbox:
        text_container.markdown(' '.join(translated_list))

    result_score = {}

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
            
            try:
                status_container.markdown('Extract result...')
                extracted_score_json = json.loads(get_json(extracted_score))
                status_container.markdown(f'')

                for t in extracted_score_json:
                    current_score = 0
                    topic_id = t["TopicID"]
                    if topic_id in result_score:
                        current_score = result_score[topic_id][0]
                    new_score = t["Score"]
                    if (new_score > current_score) or (current_score == 0):
                        result_score[topic_id] = [new_score, t["Explanation"]]

                if add_url_score_checkbox:
                    with get_openai_callback() as cb:
                        url_score = score_chain.run(topics = topics_for_prompt, article = "", url = current_url)
                    total_token_count += cb.total_tokens
                    show_total_tokens(total_token_count)
                    status_container.markdown(f'Extracted URL score...')

                    try:
                        extracted_url_score_json = json.loads(get_json(url_score))
                        status_container.markdown(f'')

                        for t in extracted_url_score_json:
                            current_score = 0
                            topic_id = t["TopicID"]
                            if topic_id in result_score:
                                current_score = result_score[topic_id][0]
                            new_score = t["Score"]
                            if (new_score > current_score) or (current_score == 0):
                                result_score[topic_id] = [new_score, t["Explanation"]]

                    except Exception as error:
                        output_container.markdown(f'Error JSON:\n\n{url_score}\n\nError: {error}\n\n{traceback.format_exc()}', unsafe_allow_html=True)
            except Exception as error:
                output_container.markdown(f'Error JSON:\n\n{extracted_score}\n\nError: {error}\n\n{traceback.format_exc()}', unsafe_allow_html=True)
        
    show_total_tokens(total_token_count)
    
    ordered_result_score = collections.OrderedDict(sorted(result_score.items()))
    result_list = []
    for s in result_score.keys():
        if s in TOPIC_DICT:
            result_list.append([TOPIC_DICT[s], *ordered_result_score[s]])
        else:
            error_container.markdown(ordered_result_score)

    bulk_row = [current_url, input_text_len, extracted_text_len, translated_lang, transpated_text_len, ordered_result_score, translated_text]
    bulk_result_list.append(bulk_row)

    if not bulk_mode_checkbox:
        df = pd.DataFrame(result_list, columns = ['Topic', 'Score', 'Explanation'])
        df = df.sort_values(by=['Score'], ascending=False)
        output_container.dataframe(df, use_container_width=True, hide_index=True)

if bulk_mode_checkbox:
    bulk_columns = ['URL', 'Input length', 'Extracted length', 'Lang', 'Translated length']
    if inc_source_checbox:
        bulk_columns.extend(['Source text'])
    for t in TOPICS_LIST:
        bulk_columns.extend([f'[{t[0]}]{t[1]}'])
        if inc_explanation_checkbox:
            bulk_columns.extend([f'[{t[0]}]Explanation'])
    bulk_data = []
    for row in bulk_result_list:
        bulk_row = [*row[:-2]]

        if inc_source_checbox:
            source_text = row[-1:][0]
            bulk_row.extend([source_text])
        
        score_data = row[-2:][0]
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
      
