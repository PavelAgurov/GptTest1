"""
    Text processing functions
"""

import tiktoken

def text_to_paragraphs(extracted_text : str,
                        token_estimator : tiktoken.core.Encoding,
                        first_paragraph_max_token : int,
                        max_tokens_translation : int) -> list[str]:
    """Split text into paragraphs by tokens"""
    result_paragraph_list = []
    extracted_sentence_list = extracted_text.split('\n')

    current_token_count = 0
    current_paragraph   = []
    for sencence in extracted_sentence_list:
        max_tokens = first_paragraph_max_token
        if len(result_paragraph_list) > 0: # first paragpath found
            max_tokens = max_tokens_translation
        token_count_p = len(token_estimator.encode(sencence))
        if current_token_count + token_count_p < max_tokens:
            current_paragraph.append(sencence)
            current_token_count = current_token_count + token_count_p
        else:
            result_paragraph_list.append('\n\n'.join(current_paragraph))
            current_paragraph  = [sencence]
            current_token_count = token_count_p
    if len(current_paragraph) > 0:
        result_paragraph_list.append('\n\n'.join(current_paragraph))
    return result_paragraph_list

def limit_text_tokens(text: str, tokenizer: tiktoken.core.Encoding, max_tokens : int) -> str:
    """Limit text by tokens"""
    input_ids = tokenizer.encode(text)
    end_index = min(max_tokens, len(input_ids))
    chunk_ids = input_ids[:end_index]
    return tokenizer.decode(chunk_ids)
