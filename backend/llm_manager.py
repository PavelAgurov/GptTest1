"""LLM Manager"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121

import os
from dataclasses import dataclass
import traceback

import langchain
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.cache import SQLiteCache
from langchain.callbacks import get_openai_callback
import tiktoken

from backend.llm import prompts
from backend.llm.refine import RefineChain
from backend.text_processing import limit_text_tokens
from backend.base_classes import TopicDefinition
from utils import get_llm_json, parse_llm_xml, str2lower

@dataclass
class LeaderRecord:
    """Leader"""
    name    : str
    company : str
    title   : str
    senior  : bool
    counter : int

@dataclass
class LeadersListResult:
    """List of leaders"""
    leaders     : list[LeaderRecord]
    used_tokens : int
    error       : str

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

class LLMManager():
    """LLM Manager"""

    callbacks: LlmCallbacks
    translation_chain : LLMChain
    score_chain : LLMChain
    llm_summary : ChatOpenAI
    leaders_chain : LLMChain
    text_splitter : CharacterTextSplitter
    token_estimator : tiktoken.core.Encoding

# model 3.5        input                 output
#4K  context  $0.0015 / 1K tokens	$0.002 / 1K tokens
#16K context  $0.003  / 1K tokens	$0.004 / 1K tokens

    MODEL_NAME = "gpt-3.5-turbo" # gpt-3.5-turbo-16k
    MAX_MODEL_TOKENS = 4097 # max token for gpt 3.5
    MAX_TOKENS_SCORE = 1600
    MAX_TOKENS_SUMMARY = 2000
    MAX_TOKENS_LEADERS = 1000
    FIRST_PARAGRAPH_MAX_TOKEN = 200 # small text to check language
    MAX_TOKENS_TRANSLATION    = 1000

    _TIKTOKEN_CACHE_DIR = ".tiktoken-cache"

    EXCLUDED_LEADER_NAMES = ['unknown', 'name of top manager', 'pmi']
    EXCLUDED_LEADER_TITLES = ['company']

    def __init__(self, open_api_key : str, callbacks: LlmCallbacks):
        self.callbacks = callbacks

        langchain.llm_cache = SQLiteCache()

        # https://github.com/openai/tiktoken/issues/75
        os.makedirs(self._TIKTOKEN_CACHE_DIR, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = self._TIKTOKEN_CACHE_DIR

        llm_translation = ChatOpenAI(
            model_name     = self.MODEL_NAME, 
            openai_api_key = open_api_key,
            max_tokens     = self.MAX_TOKENS_TRANSLATION,
            temperature    = 0,
            max_retries    = 3
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

        llm_leaders = ChatOpenAI(
            model_name     = self.MODEL_NAME, 
            openai_api_key = open_api_key, 
            max_tokens     = self.MAX_TOKENS_LEADERS,
            temperature    = 0
        )
        leaders_prompt = PromptTemplate.from_template(prompts.leaders_prompt_template)
        self.leaders_chain  = LLMChain(llm=llm_leaders, prompt = leaders_prompt)

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

    def get_token_count(self, text : str) -> int:
        """Get count of tokens in text"""
        return len(self.token_estimator.encode(text))

    def refine_text(self, text : str) -> str:
        """Create summary by refining"""
        self.report_status('Request LLM for summary...')
        text = self.clean_up_text(text)

        refine_result = RefineChain(self.llm_summary).refine(text, self.MAX_MODEL_TOKENS - self.MAX_TOKENS_SUMMARY)

        for step in refine_result.steps:
            print(step)
        print('-------------------------------------------------')
        summary = ""
        if not refine_result.error:
            summary = refine_result.summary
        else:
            self.callbacks.report_error_callback(refine_result.error)
        self.callbacks.used_tokens_callback(refine_result.tokens_used)
        self.report_status('Refining is done')
        return summary

    # def split_text_to_paragraphs(self, text : str) -> list[str]:
    #     """Split text by paragraphs"""
    #     return text_to_paragraphs(text, self.token_estimator, self.FIRST_PARAGRAPH_MAX_TOKEN, self.MAX_TOKENS_TRANSLATION)

    def translate_text(self, text : str) -> TranslationResult:
        """Translate text"""
        text = self.clean_up_text(text)
        with get_openai_callback() as cb:
            translated_text = self.translation_chain.run(input = text)
        total_tokens= cb.total_tokens
        print(translated_text)
        try:
            translated_text_json = parse_llm_xml(translated_text, ["lang", "output"])
            translated_lang = translated_text_json["lang"]
            translated_text = translated_text_json["output"]
            return TranslationResult(translated_lang, translated_text, total_tokens, None)
        except Exception as error: # pylint: disable=W0718
            print(f'Error: {error}. JSON: {translated_text}')
            return TranslationResult(None, None, total_tokens, None)
        
    def clean_up_text(self, text : str) -> str:
        """Remove dagerous chars from text"""
        return text.replace("“", "'").replace("“", "”").replace("\"", "'").replace("«", "'").replace("»", "'")

    def score_topics(self, url : str, text : str, topic_list : list[TopicDefinition]) -> ScoreTopicsResult:
        """Score all topics"""

        result_score = {}
        total_token_count = 0
        error_list = []

        current_paragraph = self.clean_up_text(text)

        topic_dict = {t.name.lower().strip() : t for t in topic_list}

        topics_for_prompt_list = []
        topics_for_prompt_list.append('<topic_list>')
        for t in topic_list:
            topics_for_prompt_list.append('<topic>')
            topics_for_prompt_list.append(f'<name>{t.name}</name>')
            topics_for_prompt_list.append(f'<description>{t.description}</description>')
            topics_for_prompt_list.append('</topic>')
        topics_for_prompt_list.append('</topic_list>')
        topics_for_prompt = "\n".join(topics_for_prompt_list)

        self.report_status('Request LLM to score...')

        prompt_without_text = self.score_chain.prompt.format(topics = topics_for_prompt, text = '')
        prompt_without_text_tokens = len(self.token_estimator.encode(prompt_without_text))
        max_tokens_score = self.MAX_MODEL_TOKENS - self.MAX_TOKENS_SCORE - prompt_without_text_tokens - 10

        extracted_score = None
        cut_information = None
        try:
            reduced_text = limit_text_tokens(current_paragraph, self.token_estimator,  max_tokens_score) # cut if needed
            reduced_text_tokens = len(self.token_estimator.encode(reduced_text))
            cut_information = f'current_paragraph size={len(current_paragraph)}), max_tokens_score={max_tokens_score}, prompt_without_text_tokens={prompt_without_text_tokens}, reduced_text_tokens={reduced_text_tokens}'
            if reduced_text != current_paragraph:
                print(f'prompt_without_text_tokens={prompt_without_text_tokens}')
                cut_information = f'{cut_information}. CUT TEXT before score: {len(current_paragraph)} => {len(reduced_text)} ({reduced_text_tokens} tokens)'
                print(cut_information)

            #prompt_full = self.score_chain.prompt.format(topics = topics_for_prompt, text = reduced_text)
            #print(prompt_full)

            with get_openai_callback() as cb:
                extracted_score = self.score_chain.run(topics = topics_for_prompt, text = reduced_text)
            total_token_count += cb.total_tokens
            self.report_status(f'Done. Got {len(extracted_score)} chars.')
            extracted_score_tokens = len(self.token_estimator.encode(extracted_score))
            print(f'extracted_score_tokens={extracted_score_tokens}')
        except Exception as error: # pylint: disable=W0718
            error_list.append(f'Error: {error}\n\n{traceback.format_exc()}. URL: {url}. {cut_information}')

        if extracted_score:
            try:
                self.report_status('Extract result...')
                extracted_score_json = get_llm_json(extracted_score)
                self.report_status('')

                for t in extracted_score_json['topics']:
                    topic_name        = t['name']
                    topic_score       = t['score']
                    topic_explanation = t['explanation']

                    topic_name_lower = str2lower(topic_name)
                    if topic_name_lower not in topic_dict:
                        error_list.append(f'Url: {url}. Topic name {topic_name} was not found in topic list')
                        continue
                    topic_item = topic_dict[topic_name_lower]
                    result_score[topic_item.id] = [topic_score, topic_explanation]

            except Exception as error: # pylint: disable=W0718
                error_list.append(f'Error:\n\n{extracted_score}\n\nError: {error}\n\n{traceback.format_exc()}. Url: {url}')
        else:
            error_list.append(f'Empty extracted_score\n\n. Url: {url}')

        self.report_status('')

        result = ScoreTopicsResult(
            total_token_count,
            '\n'.join(error_list),
            extracted_score,
            result_score
        )

        return result


    def detect_leaders(self, url : str, text : str) -> LeadersListResult:
        """Detect leaders"""
        
        # TODO - split into chunks
        reduced_text = limit_text_tokens(text, self.token_estimator,  1500)

        total_token_count = 0
        try:
            with get_openai_callback() as cb:
                extracted_leaders = self.leaders_chain.run(text = reduced_text)
            total_token_count = cb.total_tokens
        except Exception as error: # pylint: disable=W0718
            print(f'Error: {error}. URL: {url}.')
            return LeadersListResult(None, total_token_count, error)

        print(extracted_leaders)

        result = list[LeaderRecord]()
        try:
            extracted_leaders_json = get_llm_json(extracted_leaders)

            for t in extracted_leaders_json['managers']:
                leader_name = t['name']
                company     = t['company']
                title       = t['title']
                senior      = t['senior']
                counter     = t['counter']
                if title.lower() in self.EXCLUDED_LEADER_TITLES:
                    continue
                if leader_name.lower() in self.EXCLUDED_LEADER_NAMES:
                    continue
                result.append(LeaderRecord(leader_name, company, title, senior, counter))
        except Exception as error: # pylint: disable=W0718
            print(f'Error: {error}. JSON: {extracted_leaders}. URL: {url}.')
            return LeadersListResult(None, total_token_count, error)

        return LeadersListResult(result, total_token_count, None)