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
You are the best English translator. Please translate the article into English.
Please provide result in JSON format with fields:
- lang (language of original article)
- text (translated article)
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
text_container     = st.expander(label="Extracted Text")
output_container   = st.container()

header_container.markdown(how_it_work, unsafe_allow_html=True)


def get_text_from_url(url):
    response = requests.get(url)
    
    # Create a BeautifulSoup object and specify the parser library at the same time
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all the h and p tags on the page
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div'])
    
    current_prompt   = ""
    current_response = "" 
    result = []
    
    for tag in headers:
        if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
            if current_prompt and current_response:  # ensuring both prompt and response are not empty
                if len(current_response) > 20:
                    result.append(current_prompt)
                    result.append(current_response)
            current_prompt = tag.text
            current_response = ""
        elif tag.name == 'p':
            current_response += ' ' + tag.text

    # Don't forget the last one
    if current_prompt and current_response:
                result.append(current_prompt)
                result.append(current_response)

    return '\n'.join(result)

def get_json(text):
    open_bracket = text.find('[')
    if open_bracket == -1:
        return text
    close_bracket = text.find(']')
    if open_bracket == -1:
        return text
    return text[open_bracket:close_bracket+1]
     

#langchain.llm_cache = SQLiteCache()

llm = OpenAI(model_name = "text-davinci-003", max_tokens=2000)
score_prompt = PromptTemplate.from_template(score_prompt_template)
translation_prompt= PromptTemplate.from_template(translation_prompt_template)
    
#input_url = input_container.text_area("URL: ", "", key="input")
input_url = input_container.text_input("URL: ", "", key="input")


if input_url:
    debug_container.markdown('Request URL...')
    input_text = get_text_from_url(input_url)
    debug_container.markdown(f'Done. Got {len(input_text)} chars.')

    debug_container.markdown('Request LLM for translation...')
    translated_text = llm(translation_prompt.format(article = input_text))
    translated_text_json = json.loads(translated_text)
    translated_text = translated_text_json["text"]
    text_container.markdown(f'{translated_text_json["lang"]}: {translated_text}')

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
        

        
    

