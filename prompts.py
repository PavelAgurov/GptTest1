translation_prompt_template = """/
You are the best English translator. Please translate provided article (delimited with XML tags) into English.
Please provide result in JSON format with fields:
- lang (human language of original article - English, Russian, German etc.)
- translated (translated article)
Be sure that result is real JSON.

<article>{article}</article>
"""
# 

score_prompt_template = """/
You are text classification machine. 
You have numerated list of topics:
{topics}

You task is to check if each topic is relevant to the provided article (delimited with XML tags) and explain why.
Also take into considiration article's URL (delimited with XML tags) that can be also relevant or not.
URL is very important information that can bring high score for the relevant topic.

Also add score of relevance from 0 to 1 (0 - not relevant, 1 - fully relevant).
When possible you should include parts of original text to make explanation more clear.
Provide as much arguments as possible why article is related or not to the topic.
Be very scrupulous when you do classification. If it's only one or two words then it's not enough to be relevant.
When article can be considered as related to the topic, but does not provide any information - reduce score.
Exceptions are topics related to a regulation, a science or job - they can be primary even with small score.

Choose the two most relevant and least abstract topics as major and minor topics and determine their relevance.
Add explanation why you choose it as primary and secondary topics.
Do not just use previously calculated scores, think about it one more time.

Think about it step by step:
- read all topics
- read article
- generate scores for each topic
- provide output in JSON format:
{{
    "topics":[
        {{"topicID": 1, "score": 0.5, "explanation": "why article or URL are relevant or not"}},
        {{"topicID": 2, "score": 0  , "explanation": "some text here"}}
    ],
    "primary_topic":[
        "topic_id" : primary topic id,
        "score": 0.9,
        "explanation": "why this topic is primary by relevance"
    ],
    "secondary_topic":[
        "topic_id" : secondary topic id,
        "score": 0.1,
        "explanation": "why this topic is secondary by relevance"
    ]
}}

<article>{article}</article>
<article_url>{url}</article_url>
"""
