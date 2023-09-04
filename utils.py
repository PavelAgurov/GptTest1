"""
    Utils
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411

import re
import json
import traceback

def get_llm_json(text : str) -> any:
    """Get fixed LLM Json"""
    try:
        return json.loads(get_fixed_json(text))
    except Exception as error: # pylint: disable=W0718
        print('----------------------')
        print(f'Error: {error}.')
        print(f'Track: {traceback.format_exc()}')
        print(f'JSON: {text}')
        print('----------------------')
        raise error

def get_fixed_json(text : str) -> str:
    """Fix LLM json"""
    text = re.sub(r"},\s*]", "}]", text)
    open_bracket = min(text.find('['), text.find('{'))
    if open_bracket == -1:
        return text
            
    close_bracket = max(text.rfind(']'), text.rfind('}'))
    if close_bracket == -1:
        return text
    return text[open_bracket:close_bracket+1]

def grouper(iterable, step) -> list:
    """Split list into groups"""
    result = []
    for i in range(0, len(iterable), step):
        result.append(iterable[i:i+step])
    return result


