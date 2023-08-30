"""
    Text processing functions
"""

def text_extractor(footer_texts : list[str], text : str):
    """Remove footer from text"""
    for f in footer_texts:
        f = f.strip()
        if len(f) == 0:
            continue
        footer_index = text.find(f)
        if footer_index != -1:
            text = text[:footer_index]
    return text

def text_to_paragraphs(extracted_text, token_estimator, FIRST_PARAGRAPH_MAX_TOKEN, MAX_TOKENS_TRANSLATION) -> []:
    """Split text into paragraphs by tokens"""
    result_paragraph_list = []
    extracted_sentence_list = extracted_text.split('\n')

    current_token_count = 0
    current_paragraph   = []
    for p in extracted_sentence_list:
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
