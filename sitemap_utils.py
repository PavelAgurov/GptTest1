"""
    Sitemap utils
"""

import re
from dataclasses import dataclass
import urllib.request
import xml.etree.ElementTree as ET

@dataclass
class SitemapResult:
    """Result of loading sitemap"""
    url_list : list[str]
    error    : str

def __get_urlset(url : str) -> list[str]:
    with urllib.request.urlopen(url) as f:
        site_map = f.read().decode('utf-8')

    root = ET.fromstring(site_map)
    
    res = re.findall(r'\{(.*)\}(.*)', root.tag)
    schema = res[0][0]
    tag    = res[0][1]

    result = []
    if tag == 'sitemapindex':
        for url_map in root.findall(f'{{{schema}}}sitemap'):
            loc = url_map.find(f'{{{schema}}}loc').text
            result.extend(__get_urlset(loc))
        return result
    
    for url_loc in root.findall(f'{{{schema}}}url'):
        loc = url_loc.find(f'{{{schema}}}loc').text
        result.append(loc)
        
    return result

def sitemap_load(page_url : str) -> SitemapResult:
    """Load URLs from sitemap"""
    try:
        urlset = __get_urlset(page_url)
        return SitemapResult(urlset, None)
    except Exception as error:
        return SitemapResult(None, error)

