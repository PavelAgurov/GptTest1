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

refine_initial_prompt_template = """\

Write a concise summary of the text (delimited with XML tags).
Please provide result in XML format:
<summary>
    summary here
</summary>

<text>
{text}
</text>
"""

refine_combine_prompt_template = """\
Your job is to produce a final summary. We have provided an existing summary up to a certain point (delimited with XML tags).
We have the opportunity to refine the existing summary (only if needed) with some more context (delimited with XML tags).
Given the new context, refine the original summary (only if new context is useful) otherwise say that it's not useful.
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

@dataclass
class RefineResult:
    """Result of refine"""
    summary : str
    tokens_used : int
    error : str
    steps : []

class RefineChain():
    """Refine chain"""
    refine_initial_chain : LLMChain
    refine_combine_chain : LLMChain
    token_estimator : tiktoken.core.Encoding
    TOKEN_BUFFER = 150

    def __init__(self, llm : ChatOpenAI):
        if llm:
            refine_initial_prompt = PromptTemplate(template= refine_initial_prompt_template, input_variables=["text"])
            self.refine_initial_chain = LLMChain(llm= llm, prompt= refine_initial_prompt)
            refine_combine_prompt = PromptTemplate(template= refine_combine_prompt_template, input_variables=["existing_summary", "more_context"])
            self.refine_combine_chain = LLMChain(llm= llm, prompt= refine_combine_prompt)
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
            prompt_len = self.len_function(self.refine_initial_chain.prompt.format(text = ''))
            new_index = self.get_max_possible_index(
                sentence_list, 
                current_index, 
                max_tokens - prompt_len - self.TOKEN_BUFFER, 
                self.len_function
            )
            steps.append(f'Process doc {current_index}:{new_index}')
            current_doc = '.'.join(sentence_list[current_index:new_index])
            with get_openai_callback() as cb:
                summary_result = self.refine_initial_chain.run(text = current_doc)
            tokens_used += cb.total_tokens
            steps.append(summary_result)

            summary_xml = parse_llm_xml(summary_result, ["summary"])
            summary = summary_xml["summary"]
            current_index = new_index+1

            for _ in range(len(sentence_list)):
                prompt_len = self.len_function(self.refine_combine_chain.prompt.format(existing_summary = summary, more_context = ''))
                new_index = self.get_max_possible_index(
                    sentence_list, 
                    current_index, 
                    max_tokens - prompt_len - self.TOKEN_BUFFER, 
                    self.len_function
                )
                steps.append(f'Process doc {current_index}:{new_index}')
                
                refine_result = None
                try: # if not possible - we will ignore step and try next one
                    current_doc = '.'.join(sentence_list[current_index:new_index])
                    with get_openai_callback() as cb:
                        refine_result = self.refine_combine_chain.run(existing_summary = summary, more_context = current_doc)
                    tokens_used += cb.total_tokens
                    steps.append(refine_result)
                except Exception as error: # pylint: disable=W0718
                    steps.append(error)

                if refine_result:
                    refined_xml = parse_llm_xml(refine_result, ["not_useful", "refined_summary"])
                    refined_useful = not refined_xml["not_useful"]
                    if refined_useful:
                        summary = refined_xml["refined_summary"]

                current_index = new_index+1
                if new_index >= len(sentence_list):
                    break
            
            return RefineResult(summary, tokens_used, None, steps)
        except Exception as error: # pylint: disable=W0718
            steps.append(error)
            print(f'Error: {error}. Track: {traceback.format_exc()}')
            return RefineResult(summary, tokens_used, error, steps)
