import pandas as pd
import numpy as np
import json
import os
import requests
import streamlit as st
from streamlit_chat import message
import openai
import langchain
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache, SQLiteCache
from bs4 import BeautifulSoup

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

how_it_work = """\
TBD
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
You have list of topics:
[{topics}]

You task is to check if each topic is relevant to the provided article (delimited with XML tags) and explain why.
Think about it step by step.
Also add score of relevance from 0 to 1 (0 - not relevant, 1 - fully relevant).
When possible you should include parts of original text to make explanation more clear.
Provide as much arguments as possible why article is related or not to the topic.
When article can be considered as related to the topic, but does not provide any information - reduce score.
Validate all provided topics one by one.

Provide your output in json format with the keys: Topic, Score, Explanation.

Example output:
[
{{"Topic": "Tobacco Harm reduction", "Score": 0.5, "Explanation": "some text here"}},
{{"Topic": "Tobaco science", "Score": 0, "Explanation": "some text here"}},
]

<article>{article}</article>
"""

TOPICS = """
 "Tobacco Harm reduction", "Tobacco multi-product approach", "Inclusion, Diversity", "Leadership content",
  "Investor Relations", "Tobaco science", "Smoke-free vision", "Wellness, Healthcare, Health"
"""

st.set_page_config(page_title="PMI Topics Demo", page_icon=":robot:")
st.title('PMI Topics Demo')

header_container   = st.container()
input_container    = st.container()
debug_container    = st.empty()
org_text_container = st.expander(label="Original Text")
text_container     = st.expander(label="Extracted Text")
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
     

langchain.llm_cache = SQLiteCache()

llm = OpenAI(model_name = "text-davinci-003", max_tokens=2000)
score_prompt = PromptTemplate.from_template(score_prompt_template)
translation_prompt= PromptTemplate.from_template(translation_prompt_template)
    
#input_url = input_container.text_area("URL: ", "", key="input")
input_url = input_container.text_input("URL: ", "", key="input")


if input_url:
    debug_container.markdown('Request URL...')
    input_text = load_html(input_url)
    debug_container.markdown(f'Done. Got {len(input_text)} chars.')
    org_text_container.markdown(input_text)

    debug_container.markdown('Request LLM for translation...')
    translated_text = llm(translation_prompt.format(article = input_text))
    try:
        translated_text_json = get_json(translated_text)
        translated_text_json = json.loads(translated_text_json)
        translated_lang = translated_text_json["lang"]
        if translated_lang == "English":
            text_container.markdown("No translation")
        else:
            translated_text = translated_text_json["translated"]
            text_container.markdown(f'{translated_lang}: {translated_text}')
    except:
        text_container.markdown(translated_text)

    debug_container.markdown('Request LLM to score...')
    extracted_score = llm(score_prompt.format(topics = TOPICS, article = translated_text))
    extracted_score = get_json(extracted_score)
    debug_container.markdown(f'Done. Got {len(extracted_score)} chars.')

    result = []
    try:
        debug_container.markdown('Extract result...')
        extracted_score_json = json.loads(f'{extracted_score}')
        debug_container.markdown(f'')
       
        for t in extracted_score_json:
            result.append([t["Topic"], t["Score"], t["Explanation"]])
            
        df = pd.DataFrame(result, columns = ['Topic', 'Score', 'Explanation'])
        df = df.sort_values(by=['Score'], ascending=False)
        output_container.markdown(df.to_html(index=False), unsafe_allow_html=True)

    except Exception as error:
        output_container.markdown(f'Error JSON: {extracted_score}. Error: {error}', unsafe_allow_html=True)
        

        
    

