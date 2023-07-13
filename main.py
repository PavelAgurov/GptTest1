import pandas as pd
import json
import os
import streamlit as st
import langchain
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.cache import SQLiteCache
import traceback

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

how_it_work = """\
Insert URL here and check output. Only one URL supported now.
"""

translation_prompt_template = """/
You are the best English translator. Please translate provided article into English.
Please provide result in JSON format with fields:
- lang (language of original article)
- translated (translated article)
Be sure that result is real JSON.
Article: {article}
"""


score_prompt_template = """/
You are text classification machine. 
You have numerated list of topics:
{topics}

You task is to check if each topic is relevant to the provided article (delimited with XML tags) and explain why.
Think about it step by step.
Also add score of relevance from 0 to 1 (0 - not relevant, 1 - fully relevant).
When possible you should include parts of original text to make explanation more clear.
Provide as much arguments as possible why article is related or not to the topic.
When article can be considered as related to the topic, but does not provide any information - reduce score.
Validate all provided topics one by one.

Provide your output in json format with the keys: TopicID, Topic, Score, Explanation.

Example output:
[
{{"TopicID": 1, "Topic": "Tobacco Harm reduction", "Score": 0.5, "Explanation": "some text here"}},
{{"TopicID": 2, "Topic": "Tobaco science", "Score": 0, "Explanation": "some text here"}},
]

<article>{article}</article>
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

tab_one, tab_bulk = st.tabs(["Process one URL", "Bulk processing"])

with tab_one:
    header_container   = st.container()
    input_container    = st.container()
    debug_container    = st.empty()
    org_text_container = st.expander(label="Original Text")
    text_container     = st.expander(label="Extracted (and translated) Text")
    output_container   = st.container()

header_container.markdown(how_it_work, unsafe_allow_html=True)

from unstructured.partition.html import partition_html

def load_html(url):
  elements = partition_html(url=url)
  return "\n\n".join([str(el) for el in elements])


def get_json(text):
    open_bracket = text.find('[')
    if open_bracket == -1:
        open_bracket = text.find('{')
        if open_bracket == -1:
            return text
            
    close_bracket = text.find(']')
    if close_bracket == -1:
        close_bracket = text.find('}')
        if close_bracket == -1:
            return text
    return text[open_bracket:close_bracket+1]

def grouper(iterable, step):
    result = []
    for i in range(0, len(iterable), step):
        result.append(iterable[i:i+step])
    return result

langchain.llm_cache = SQLiteCache()

llm_score = OpenAI(model_name = "text-davinci-003", max_tokens = 2000)
llm_trans = OpenAI(model_name = "gpt-3.5-turbo"   , max_tokens = 2000)


score_prompt = PromptTemplate.from_template(score_prompt_template)
translation_prompt= PromptTemplate.from_template(translation_prompt_template)
    
#input_url = input_container.text_area("URL: ", "", key="input")
input_url = input_container.text_input("URL: ", "", key="input")


# Long English: https://www.pmi.com/us/about-us/our-leadership-team
# Non English https://www.pmi.com/markets/portugal/pt/news/details?articleId=tabaqueira-participa-na-consulta-pública-para-a-estratégia-nacional-de-luta-contra-o-cancro-2021-2030

topic_chunks = grouper(TOPICS_LIST, 3)

if input_url:
    debug_container.markdown('Request URL...')
    input_text = load_html(input_url)
    debug_container.markdown(f'Done. Got {len(input_text)} chars.')
    org_text_container.markdown(input_text)

    input_text_list = input_text.split('\n')

    first_paragpaph_size = 200 # small text to check language
    next_paragpaph_size  = 800

    paragpaph_list = []
    current_word_count = 0
    current_paragpaph  = []
    for i, p in enumerate(input_text_list):
        words_count_p = len(p.split(' '))
        current_word_count = current_word_count + words_count_p
        size = first_paragpaph_size
        if i > 0:
            size = next_paragpaph_size
        if (current_word_count < size):
            current_paragpaph.append(p)
        else:
            paragpaph_list.append('\n\n'.join(current_paragpaph))
            current_paragpaph  = []
            current_word_count = 0
    if len(current_paragpaph) > 0:
        paragpaph_list.append('\n\n'.join(current_paragpaph))

    translated_list = []
    no_translation = False
    for i, p in enumerate(paragpaph_list):
        debug_container.markdown(f'Request LLM for translation {i+1}/{len(paragpaph_list)}...')
        translated_text = llm_trans(translation_prompt.format(article = p))
        try:
            translated_text_json = json.loads(get_json(translated_text))
            translated_lang = translated_text_json["lang"]
            if translated_lang == "English":
                no_translation = True
                debug_container.markdown(f'Text is in English. No translation needed.')
                break
            translated_text = translated_text_json["translated"]
            translated_list.append(translated_text)
        except:
            no_translation = True

    if not no_translation:
        text_container.markdown(' '.join(translated_list))
    else:
        text_container.markdown("Text is in English. No translation needed.")
        translated_list = paragpaph_list # just use "as is"
        
    result_score = {}
    
    for i, p in enumerate(translated_list):
        for j, topic_def in enumerate(topic_chunks):
            topics_for_prompt = "\n".join([f'{t[0]}. {t[2]}' if len(t[2])>0 else f'{t[0]}. {t[1]}' for t in topic_def])
            topics_id_name_list = {t[0]:t[1] for t in topic_def}

            debug_container.markdown(f'Request LLM to score {i+1}/{len(translated_list)}, topics chunk {j+1}/{len(topic_chunks)}...')
            extracted_score = llm_score(score_prompt.format(topics = topics_for_prompt, article = p))
            extracted_score = get_json(extracted_score)
            debug_container.markdown(f'Done. Got {len(extracted_score)} chars.')

            try:
                debug_container.markdown('Extract result...')
                extracted_score_json = json.loads(f'{extracted_score}')
                debug_container.markdown(f'')
            
                for t in extracted_score_json:
                    current_score = 0
                    topic_name = topics_id_name_list[t["TopicID"]]
                    if topic_name in result_score:
                        current_score = result_score[topic_name][0]
                    new_score = t["Score"]
                    if (new_score > current_score) or (current_score == 0):
                        result_score[topic_name] = [new_score, t["Explanation"]]

            except Exception as error:
                output_container.markdown(f'Error JSON: {extracted_score}. Error: {error}\n{traceback.format_exc()}', unsafe_allow_html=True)
        
    result_list = []
    for s in result_score.keys():
        result_list.append([s, *result_score[s]])

    df = pd.DataFrame(result_list, columns = ['Topic', 'Score', 'Explanation'])
    df = df.sort_values(by=['Score'], ascending=False)
    output_container.markdown(df.to_html(index=False), unsafe_allow_html=True)

      