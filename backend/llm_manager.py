"""LLM Manager"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121

from dataclasses import dataclass
import traceback

import langchain
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
from langchain.callbacks import get_openai_callback
import tiktoken

import backend.llm.prompts as prompts
from backend.llm.refine import RefineChain
from backend.text_processing import text_to_paragraphs
from backend.base_classes import TopicDefinition
from utils import get_llm_json

@dataclass
class LlmCallbacks:
    """Set of callbacks"""
    report_status_callback : any
    used_tokens_callback   : any
    report_error_callback  : any

@dataclass
class TranslationResult:
    """Result of translation"""
    lang        : str
    translation : str
    used_tokens : int
    error       : str

@dataclass
class ScoreTopicsResult:
    """Result of score"""
    used_tokens          : int
    error                : str
    debug_json_score     : str
    result_score         : {}
    primary_topic_json   : {}
    secondary_topic_json : {}

class LLMManager():
    """LLM Manager"""

    callbacks: LlmCallbacks
    translation_chain : LLMChain
    score_chain : LLMChain
    llm_summary : ChatOpenAI
    text_splitter : CharacterTextSplitter
    token_estimator : tiktoken.core.Encoding

    MODEL_NAME = "gpt-3.5-turbo"
    MAX_MODEL_TOKENS = 4097 # max token for gpt 3.5
    MAX_TOKENS_SCORE = 2000
    MAX_TOKENS_SUMMARY = 2500
    FIRST_PARAGRAPH_MAX_TOKEN = 200 # small text to check language
    MAX_TOKENS_TRANSLATION    = 1000

    def __init__(self, open_api_key : str, callbacks: LlmCallbacks):
        self.callbacks = callbacks

        langchain.llm_cache = SQLiteCache()

        llm_translation = ChatOpenAI(
            model_name     = self.MODEL_NAME, 
            openai_api_key = open_api_key,
            max_tokens     = self.MAX_TOKENS_TRANSLATION,
            temperature    = 0
        )
        translation_prompt = PromptTemplate.from_template(prompts.translation_prompt_template)
        self.translation_chain = LLMChain(llm=llm_translation, prompt = translation_prompt)

        llm_score = ChatOpenAI(
            model_name = self.MODEL_NAME,
            openai_api_key = open_api_key,
            max_tokens = self.MAX_TOKENS_SCORE,
            temperature = 0
        )
        score_prompt = PromptTemplate.from_template(prompts.score_prompt_template)
        self.score_chain  = LLMChain(llm=llm_score, prompt = score_prompt)

        self.llm_summary = ChatOpenAI(
            model_name     = self.MODEL_NAME, 
            openai_api_key = open_api_key, 
            max_tokens     = self.MAX_TOKENS_SUMMARY,
            temperature    = 0
        )

        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name= self.MODEL_NAME, 
            model_name= self.MODEL_NAME, 
            chunk_size=1000, 
            chunk_overlap=0
        )

        self.token_estimator = tiktoken.encoding_for_model(self.MODEL_NAME)

    def report_status(self, status_str : str):
        """Report status"""
        self.callbacks.report_status_callback(status_str)

    def refine_text(self, text : str) -> str:
        """Create summary by refining"""
        self.report_status('Request LLM for summary...')
        refine_result = RefineChain(self.llm_summary).refine(text, self.MAX_MODEL_TOKENS - self.MAX_TOKENS_SUMMARY)
        print(refine_result.steps)
        summary = ""
        if not refine_result.error:
            summary = refine_result.summary
        else:
            self.callbacks.report_error_callback(refine_result.error)
        self.callbacks.used_tokens_callback(refine_result.tokens_used)
        self.report_status('Refining is done')
        return summary

    def text_to_paragraphs(self, text : str) -> list[str]:
        """Split text by paragraphs"""
        return text_to_paragraphs(text, self.token_estimator, self.FIRST_PARAGRAPH_MAX_TOKEN, self.MAX_TOKENS_TRANSLATION)

    def translate_text(self, text : str) -> TranslationResult:
        """Translate text"""
        with get_openai_callback() as cb:
            translated_text = self.translation_chain.run(article = text)
        total_tokens= cb.total_tokens
        try:
            translated_text_json = get_llm_json(translated_text)
            translated_lang = translated_text_json["lang"]
            translated_text = translated_text_json["translated"]
            return TranslationResult(translated_lang, translated_text, total_tokens, None)
        except Exception as error: # pylint: disable=W0718
            print(f'Error: {error}. JSON: {translated_text}')
            return TranslationResult(None, None, total_tokens, None)
        
    def clean_up(self, text):
        """Remove dagerous chars from text"""
        return text.replace("“", "'").replace("“", "”").replace("\"", "'").replace("«", "'").replace("»", "'")

    def score_topics(self, url : str, paragraph_list : list[str], topic_chunks : list[list[TopicDefinition]]) -> ScoreTopicsResult:
        """Score all topics"""

        result_score = {}
        result_primary_topic_json   = {}
        result_secondary_topic_json = {}
        debug_json_score = []
        total_token_count = 0
        error_list = []

        for i, p in enumerate(paragraph_list):
            for j, topic_def in enumerate(topic_chunks):
                topics_for_prompt = "\n".join([f'{t.id}. {t.description}' for t in topic_def])

                self.report_status(f'Request LLM to score paragraph: {i+1}/{len(paragraph_list)}, topics chunk: {j+1}/{len(topic_chunks)}...')
                try:
                    with get_openai_callback() as cb:
                        extracted_score = self.score_chain.run(topics = topics_for_prompt, article = p, url = url)
                    total_token_count += cb.total_tokens
                    self.report_status(f'Done. Got {len(extracted_score)} chars.')
                    debug_json_score.append(extracted_score)

                    self.report_status('Extract result...')
                    extracted_score_json = get_llm_json(extracted_score)
                    self.report_status('')

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

                except Exception as error: # pylint: disable=W0718
                    error_list.append(f'Error:\n\n{extracted_score}\n\nError: {error}\n\n{traceback.format_exc()}')
            
        self.report_status('')

        result = ScoreTopicsResult(
            total_token_count,
            '\n'.join(error_list),
            '\n'.join(debug_json_score),
            result_score,
            result_primary_topic_json,
            result_secondary_topic_json
        )

        return result
