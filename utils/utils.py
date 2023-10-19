"""
    Utils
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,W1203

import re
import json
import traceback
import logging

logger : logging.Logger = logging.getLogger()

def str2lower(text : str, none_value : str = None) -> str:
    """Lower case of text"""
    if not text:
        return none_value
    return text.lower().strip()

def parse_llm_xml(text : str, variables : list[str]) -> dict[str, str]:
    """Parse XML for LLM"""
    result = dict[str, str]()
    for var_name in variables:
        result[var_name] = ''
        start_var_name = f'<{var_name}>'
        start_index = text.find(start_var_name)
        if start_index == -1:
            logger.error(f'LLM xml has no open tag variable: {var_name}')
            if len(variables) == 1:
                result[var_name] = text
            continue
        end_index = text.find(f'</{var_name}>')
        if end_index == -1:
            logger.error(f'LLM xml has no close tag variable: {var_name}')
            # hack - if we have only one tag and it's not closed - guess that it's full text
            if len(variables) == 1:
                var_value_fixed = text[start_index + len(start_var_name):]
                if var_value_fixed:
                    var_value_fixed = var_value_fixed.strip()
                result[var_name] = var_value_fixed
            continue
        var_value = text[start_index + len(start_var_name):end_index]
        if var_value:
            var_value = var_value.strip()
        result[var_name] = var_value
    return result


def get_llm_json(text : str) -> any:
    """Get fixed LLM Json"""
    try:
        return json.loads(get_fixed_json(text))
    except Exception as error: # pylint: disable=W0718
        logger.error('----------------------')
        logger.error(f'Error: {error}.')
        logger.error(f'Track: {traceback.format_exc()}')
        logger.error(f'JSON: {text}')
        logger.error('----------------------')
        raise error

def get_fixed_json(text : str) -> str:
    """Fix LLM json"""
    text = re.sub(r"},\s*]", "}]", text)
    text = re.sub(r"}\s*{", "},{", text)

    open_bracket = min(text.find('['), text.find('{'))
    if open_bracket == -1:
        return text
            
    close_bracket = max(text.rfind(']'), text.rfind('}'))
    if close_bracket == -1:
        return text
    return text[open_bracket:close_bracket+1]

