"""
    Classes to read from HTML per domain
"""


HTML_CLASSES_WHITELIST = {
    'www.pmi.com': ['content-block', 'is-content', 'entry--description', 'quote-', 'written-by']
}

HTML_CLASSES_BLACKLIST = {
    'www.unsmoke.co.za': ['-footer', '-header', 'cmp-nav_', '-banner', 'age-gate--active']
}
