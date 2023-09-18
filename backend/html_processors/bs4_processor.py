"""
    Process HTML with BS4
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from bs4 import BeautifulSoup

def get_plain_text_bs4(html : str) -> str:
    """Plain text based on BS4"""
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(['p', 'div'])
    paragraph_list = []
    for t in texts:
        paragraph = t.get_text().strip()
        if len(paragraph) == 0:
            continue
        lines = paragraph.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(' ')
            if len(words) < 2:
                continue
            paragraph_list.append(line)
    return "\n\n".join(paragraph_list)