"""
    Utils
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411

from unstructured.partition.html import partition_html
import re

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

def load_html(url : str) -> str:
    """Load HTML"""
    elements = partition_html(url=url)
    return "\n\n".join([str(el) for el in elements])

def grouper(iterable, step) -> list:
    """Split list into groups"""
    result = []
    for i in range(0, len(iterable), step):
        result.append(iterable[i:i+step])
    return result

def sort_dict_by_value(d, reverse = False):
    """Sort dict"""
    return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

def num_tokens_from_string(string, llm_encoding):
    """Get count of tokens for string"""
    return len(llm_encoding.encode(string))
