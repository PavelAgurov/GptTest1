"""
    Sitemap utils
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121

import re
from dataclasses import dataclass
import urllib.request
import xml.etree.ElementTree as ET

@dataclass
class SitemapResult:
    """Result of loading sitemap"""
    url_list : list[str]
    error    : str

def __is_excluded(url : str, excluded_prefix_list : list[str], exluded_urls_list : list[str]) -> bool:
    """Is URL excluded?"""
    for excluded_prefix_item in excluded_prefix_list:
        if url.startswith(excluded_prefix_item):
            return True
        
    for excluded_url_item in exluded_urls_list:
        excluded_url_item = excluded_url_item.strip('\\').strip('/').lower()
        if url.lower() == excluded_url_item:
            return True
    
    return False

def __get_urlset(url : str, excluded_prefix_list : list[str], exluded_urls_list : list[str]) -> list[str]:
    """"Read sitemap"""
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
            result.extend(__get_urlset(loc, excluded_prefix_list, exluded_urls_list))
        return result
    
    for url_loc in root.findall(f'{{{schema}}}url'):
        loc = url_loc.find(f'{{{schema}}}loc').text
        if not __is_excluded(loc, excluded_prefix_list, exluded_urls_list):
            result.append(loc)
        
    return result

def sitemap_load(page_url : str, excluded_prefix_list : list[str], exluded_urls_list : list[str]) -> SitemapResult:
    """Load URLs from sitemap"""
    try:
        urlset = __get_urlset(page_url, excluded_prefix_list, exluded_urls_list)
        return SitemapResult(urlset, None)
    except Exception as error: # pylint: disable=W0718
        return SitemapResult(None, error)

