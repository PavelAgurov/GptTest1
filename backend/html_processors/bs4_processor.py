"""
    Process HTML with BS4
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0902,R0903

from bs4 import BeautifulSoup, Tag

def need_to_parse_whitelist(t : Tag, html_classes_whitelist : list[str]) -> bool:
    """True if tag should be parsed"""
    if not html_classes_whitelist: # no limitation defined
        return True
    
    if not t.attrs or 'class' not in t.attrs: # we have limitation, but no class attr
        return False
    
    tag_classes = t.attrs['class']
    for html_class in html_classes_whitelist:
        for tag_classe in tag_classes:
            if html_class in tag_classe: # yes, this class shold be parsed
                return True
    return False

def need_to_parse_blacklist(t : Tag, html_classes_blacklist : list[str]) -> bool:
    """True if tag should be parsed"""
    if not html_classes_blacklist: # no limitation defined
        return True
    
    if not t.attrs or 'class' not in t.attrs: # we have limitation, but no class attr
        return False
    
    tag_classes = t.attrs['class']
    for html_class in html_classes_blacklist:
        for tag_classe in tag_classes:
            if html_class in tag_classe: # this class shold NOT be parsed
                return False
    return True

def get_plain_text_bs4(html : str, html_classes_whitelist : list[str], html_classes_blacklist : list[str]) -> str:
    """Plain text based on BS4"""

    print(f'html_classes_whitelist={html_classes_whitelist}')
    print(f'html_classes_blacklist={html_classes_blacklist}')

    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(['p', 'div', 'blockquote'])
    paragraph_list = []
    for t in texts:
        if not t:
            continue
            
        # limitation defined
        if html_classes_whitelist: 
            if need_to_parse_whitelist(t, html_classes_whitelist):
                paragraph_list.append(t.get_text().strip())
            continue

        # limitation defined
        if html_classes_blacklist: 
            if not need_to_parse_blacklist(t, html_classes_blacklist):
                continue

        # no limitation - try to extract what possible
        paragraph = t.get_text(separator=" ", strip=True).strip()
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