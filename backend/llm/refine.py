"""
    Refine summary
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903

from dataclasses import dataclass
import tiktoken
import traceback
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

from utils import parse_llm_xml

@dataclass
class RefineResult:
    """Result of refine"""
    summary     : str
    tokens_used : int
    error       : str
    steps       : list[str]

refine_initial_prompt_template = """\

Write a concise summary of the text (delimited with XML tags).
Try to extract as much as possible useful information from provided text.
If the text does not contain any information to summarize say "No summary".

Please provide result in XML format:
<summary>
Summary here
</summary>

<text>
{text}
</text>
"""

refine_combine_prompt_template = """\
Your job is to produce a final summary. We have provided an existing summary up to a certain point (delimited with XML tags).
We have the opportunity to refine the existing summary (only if needed) with some more context (delimited with XML tags).
Given the new context, refine the original summary (only if new context is useful) otherwise say that it's not useful.
Try to extract as much as possible information from new context.

Please provide result in XML format:
<not_useful>
    True if new context was not useful, False if new content was used
</not_useful>
<refined_summary>
    Refined summary here if new context was useful
</refined_summary>

<existing_summary>
{existing_summary}
</existing_summary>

<more_context>
{more_context}
</more_context>
"""

facts_prompt_template = """\
You are professional linguist. 
Your task is to extract all the information about people from the provided text (delimited with XML tags).
Who are they? What does the text say about them? What role do they play in the text?
Do not make up information, use only information from provided text.
If the text does not contain any facts just say "No facts".

Please provide result in XML format:
<personal_information>
Extracted facts about persons.
</personal_information>

<text>
{text}
</text>
"""

@dataclass
class RefineInitialResult:
    """Result of initial refine"""
    summary     : str
    tokens_used : int
    steps       : list[str]

@dataclass
class RefineStepResult:
    """Result of refine step"""
    new_summary : str
    tokens_used : int
    useful      : bool
    steps       : list[str]

@dataclass
class FactExtractionResult:
    """Result of fact extraction"""
    facts       : str
    tokens_used : int
    steps       : list[str]

class RefineChain():
    """Refine chain"""
    refine_initial_chain : LLMChain
    refine_combine_chain : LLMChain
    facts_chain          : LLMChain
    token_estimator : tiktoken.core.Encoding
    TOKEN_BUFFER = 150

    def __init__(self, llm : ChatOpenAI):
        if llm:
            refine_initial_prompt = PromptTemplate(template= refine_initial_prompt_template, input_variables=["text"])
            self.refine_initial_chain = LLMChain(llm= llm, prompt= refine_initial_prompt)
            
            refine_combine_prompt = PromptTemplate(template= refine_combine_prompt_template, input_variables=["existing_summary", "more_context"])
            self.refine_combine_chain = LLMChain(llm= llm, prompt= refine_combine_prompt)

            fact_prompt = PromptTemplate(template= facts_prompt_template, input_variables=["text"])
            self.facts_chain = LLMChain(llm= llm, prompt= fact_prompt)

            self.token_estimator = tiktoken.encoding_for_model(llm.model_name)

    def len_function(self, text : str) -> int:
        """Lenght function"""
        return len(self.token_estimator.encode(text))

    def get_max_possible_index(self, sentence_list : list[str], start_index : int, max_tokens : int, len_function : any) -> list[str]:
        """Find next possible part of text"""
        token_count = 0
        for sentence_index in range(start_index, len(sentence_list)):
            token_count_p = len_function(sentence_list[sentence_index])
            token_count = token_count + token_count_p
            if token_count <= max_tokens:
                continue
            return sentence_index
        return len(sentence_list)
    
    def refine(self, text : str, max_tokens : int) -> RefineResult:
        """Refine call"""
        sentence_list = text.split('.')

        tokens_used = 0
        summary = ""
        steps = []

        try:
            current_index = 0
            summary_step  = True
            for _ in range(len(sentence_list)+1):

                # execute first step -  summary
                if summary_step:
                    prompt_len = self.len_function(self.refine_initial_chain.prompt.format(text = ''))
                    new_index = self.get_max_possible_index(
                        sentence_list, 
                        current_index, 
                        max_tokens - prompt_len - self.TOKEN_BUFFER, 
                        self.len_function
                    )
                    steps.append(f'--- Process doc {current_index}:{new_index} / {len(sentence_list)}')
                    current_doc = '.'.join(sentence_list[current_index:new_index])

                    refine_initial_result = self.execute_initial_refine(current_doc)
                    tokens_used += refine_initial_result.tokens_used
                    steps.extend(refine_initial_result.steps)
                    summary = refine_initial_result.summary

                    # # fact extraction from first summary
                    # fact_extraction_result = self.execute_fact_extraction(current_doc)
                    # tokens_used += fact_extraction_result.tokens_used
                    # steps.extend(fact_extraction_result.steps)

                    current_index = new_index+1
                    if new_index >= len(sentence_list):
                        break

                    if not summary: # wait for valuable summary first
                        continue

                    summary_step = False
                    continue

                # execute refine
                prompt_len = self.len_function(self.refine_combine_chain.prompt.format(existing_summary = summary, more_context = ''))
                new_index = self.get_max_possible_index(
                    sentence_list, 
                    current_index, 
                    max_tokens - prompt_len - self.TOKEN_BUFFER, 
                    self.len_function
                )
                steps.append(f'--- Process doc {current_index}:{new_index}')

                current_doc = '.'.join(sentence_list[current_index:new_index])

                # # fact extraction from first summary
                # fact_extraction_result = self.execute_fact_extraction(current_doc)
                # tokens_used += fact_extraction_result.tokens_used
                # steps.extend(fact_extraction_result.steps)

                refine_step_result = self.execute_refine_step(summary, current_doc)
                tokens_used += refine_step_result.tokens_used
                steps.extend(refine_step_result.steps)
                if refine_step_result.useful:
                    summary = refine_step_result.new_summary

                current_index = new_index+1
                if new_index >= len(sentence_list):
                    break
            
            return RefineResult(summary, tokens_used, None, steps)
        except Exception as error: # pylint: disable=W0718
            steps.append(error)
            print(f'Error: {error}. Track: {traceback.format_exc()}')
            return RefineResult(summary, tokens_used, error, steps)

    def execute_initial_refine(self, document : str) -> RefineInitialResult:
        """Execute refine step"""
        tokens_used    = 0
        steps          = list[str]()
        summary        = ''

        refine_initial_result  = None
        try:
            with get_openai_callback() as cb:
                refine_initial_result = self.refine_initial_chain.run(text = document)
            tokens_used = cb.total_tokens
            steps.append(refine_initial_result)
        except Exception as error: # pylint: disable=W0718
            steps.append(error)

        if refine_initial_result:
            summary_xml = parse_llm_xml(refine_initial_result, ["summary"])
            summary_str = summary_xml["summary"].strip()
            if "no summary" not in summary_str.lower():
                summary = summary_str
        return RefineInitialResult(summary, tokens_used, steps)

    def execute_refine_step(self, existing_summary : str, more_context : str) -> RefineStepResult:
        """Execute refine step"""
        tokens_used    = 0
        refined_useful = False
        steps          = list[str]()
        summary        = ''

        refine_step_result  = None
        try:
            with get_openai_callback() as cb:
                refine_step_result = self.refine_combine_chain.run(existing_summary = existing_summary, more_context = more_context)
            tokens_used = cb.total_tokens
            steps.append(refine_step_result)
        except Exception as error: # pylint: disable=W0718
            steps.append(error)

        if refine_step_result:
            refined_xml = parse_llm_xml(refine_step_result, ["not_useful", "refined_summary"])
            refined_useful = not refined_xml["not_useful"]
            if refined_useful:
                summary = refined_xml["refined_summary"]

        return RefineStepResult(summary, tokens_used, refined_useful, steps)


    def execute_fact_extraction(self, document : str) -> FactExtractionResult:
        """Run fact extraction"""
        tokens_used = 0
        facts       = ''
        steps       = list[str]()

        # fact extraction
        fact_result = None
        try:
            with get_openai_callback() as cb:
                fact_result = self.facts_chain.run(text = document)
            tokens_used += cb.total_tokens
        except Exception as error: # pylint: disable=W0718
            steps.append(error)

        if fact_result:
            fact_xml = parse_llm_xml(fact_result, ["personal_information"])
            fact_str = fact_xml["personal_information"]
            steps.append(fact_str)
            if "no facts" not in fact_str:
                facts = fact_str

        return FactExtractionResult(facts, tokens_used, steps)
