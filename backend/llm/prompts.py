"""
    Prompts for LLM
"""

# pylint: disable=C0103,C0301

translation_prompt_template = """/
You are the best English translator. Please translate provided article (delimited with XML tags) into English.
If article is in English already - just say that it's in English already and do not translate it.
Do not return original article.

Please provide result in XML format:
<lang>
Human language of original article - English, Russian, German etc. Put only language name here.
</lang>

<output>
Translated article if language of provided article was not English.
</output>

<input>{input}</input>
"""

score_prompt_template = """\
You are text classifier. There list of topics (delimited with XML tags) and text (delimited with XML tags).
Your task is to read all text, calculate a score of relevance between this text and provided topics one by one and than sort the topics by the relevance.
A score is a double-precision floating point number (0.00 - not relevant at all, 1.00 - fully relevant).
Also you should add a brief explanation of the score.

If the text does not mention anything related to the topic - score should be 0.

Do not make up new topics, use only provided topic list!

Provide the output in JSON format.

{topics}

<text>
{text}
</text>
"""

leaders_prompt_template = """\
You are press secretary.
Your task is to extract all managers (president, head, chief, CEO, VP, SVP etc.) and their titles from provided text (delimited with XML tags).
Do not make up answers, use only provided text!

Provide the output in JSON:

"managers":{{
    {{
        "name" : "name of top manager",
        "company" : "company name where he/she is working or worked",
        "title" : "title of top manager",
        "senior": True, if manager is a senior staff,
        "counter" : how many times name appears in the text
    }}
}}

<text>
{text}
</text>
"""