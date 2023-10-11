"""
    URL patters with fixed topic
"""

from backend.base_classes import FixedTopicPattern


FIXED_TOPIC_PATTERNS : list[FixedTopicPattern] = [
    FixedTopicPattern(
        'https://www.pmi.com/investor-relations/overview/event-details?eventId=',
        'Investor Relations', 
        '',
        True
    ),
    FixedTopicPattern(
        'https://www.pmi.com/careers/job-details?id=',
        'Jobs',
        '',
        True
    ),
    FixedTopicPattern(
        'https://www.pmi.com/investor-relations/',
        'Investor Relations',
        '',
        False
    ),
    FixedTopicPattern.Ignore('https://www.pmi.com/protected-area'),
    FixedTopicPattern.Ignore('https://www.pmi.com/markets/egypt/ar')
]

FIXED_TOPIC_PATTERNS_DICT = {f.url_prefix: f for f in FIXED_TOPIC_PATTERNS}
