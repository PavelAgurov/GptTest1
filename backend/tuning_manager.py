"""
    Tuning manager
"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903

import pandas as pd

from backend.base_classes import TopicDefinition

class TuningManager():
    """Tuning Manager"""

    COLUMN_SOURCE_TEXT = 'Source text'
    COLUMN_PRIMARY     = 'Primary'
    
    def get_tuning_prompt(self, bulk_data : pd.DataFrame, topic_list : list[TopicDefinition]) -> str:
        """Get tuning prompt"""

        if bulk_data is None:
            return "No data"
        
        if self.COLUMN_SOURCE_TEXT not in bulk_data.columns:
            return "No source data column"

        if self.COLUMN_PRIMARY not in bulk_data.columns:
            return "No primary column"

        prompt_begin = """\
            I have list of texts and assigned topic for each text. 

            I want you to do following steps:
            - read all texts
            - read all topic descriptions
            - calculate score of relevance of topic for each text
            - check if your answer is equal to provided
            - correct descriptions of each topic to make descriptions the most relevant to the provided text. Description should be short and clear.
            - check score again with new descriptions
            - repeat description correction until you will have score 1.00 for right topic and close to 0.00 for other.

            Output will be:
            - corrected topic descriptions
            - new score of each text to provided topic
        """
        prompt_begin = '\n'.join([s.strip() for s in prompt_begin.split('\n')])

        prompt_topics = []
        prompt_topics.append("<topic_list>")
        for t in topic_list:
            prompt_topics.append("<topic>")
            prompt_topics.append(f"<name>{t.name}</name>")
            prompt_topics.append(f"<description>{t.description}</description>")
            prompt_topics.append("</topic>")
        prompt_topics.append("</topic_list>")
        prompt_topics_str = '\n'.join(prompt_topics)

        data = bulk_data.loc[:, [self.COLUMN_PRIMARY, self.COLUMN_SOURCE_TEXT]]
        source_text = []
        for data_row in data.values:
            source_text.append(f'Assigned topic [{data_row[0]}]. Text: {data_row[1]}')
            source_text.append('\n')
        source_text_str = '\n'.join(source_text)

        return f'{prompt_begin}\n\n{prompt_topics_str}\n\n{source_text_str}'
