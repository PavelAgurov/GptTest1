"""Bulk output"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903

import pandas as pd
from dataclasses import dataclass

from backend.base_classes import ScoreResultItem

@dataclass
class BulkOutputParams:
    """Parameters for create output data"""
    inc_explanation : bool
    inc_source      : bool

class BulkOutput():
    """Class to build output data"""

    def create_data(self, topic_list : any, bulk_result : list[ScoreResultItem], params : BulkOutputParams) -> pd.DataFrame:
        """Create output data"""
        bulk_columns = ['URL', 'Input length', 'Extracted length', 'Lang', 'Translated length']
        bulk_columns.extend(['Primary', 'Primary score'])
        if params.inc_explanation:
            bulk_columns.extend(['Primary explanation'])
        bulk_columns.extend(['Secondary', 'Secondary score'])
        if params.inc_explanation:
            bulk_columns.extend(['Secondary explanation'])
        if params.inc_source:
            bulk_columns.extend(['Source text'])
        for topic_item in topic_list:
            bulk_columns.extend([f'[{topic_item[0]}]{topic_item[1]}'])
            if params.inc_explanation:
                bulk_columns.extend([f'[{topic_item[0]}]Explanation'])

        bulk_data = []
        for row in bulk_result:
            if row.page_not_found:
                erorr_row = [row.current_url, "Page not found"]
                erorr_row.extend([None]*(len(bulk_columns)-2))
                bulk_data.append(erorr_row)
                continue

            bulk_row = []
            bulk_row.extend([row.current_url, row.input_text_len, row.extracted_text_len, row.translated_lang, row.transpated_text_len])

            bulk_row.extend([row.main_topics.primary.topic, row.main_topics.primary.topic_score]) # primary topic
            if params.inc_explanation:
                bulk_row.extend([row.main_topics.primary.explanation])

            bulk_row.extend([row.main_topics.secondary.topic, row.main_topics.secondary.topic_score]) # secondary topic
            if params.inc_explanation:
                bulk_row.extend([row.main_topics.secondary.explanation])

            if params.inc_source:
                bulk_row.extend([row.full_translated_text])
            
            score_data = row.ordered_result_score
            for topic_item in topic_list:
                if topic_item[0] in score_data:
                    topic_score = score_data[topic_item[0]]
                    bulk_row.extend([topic_score[0]])
                    if params.inc_explanation:
                        bulk_row.extend([topic_score[1]])
                else:
                    bulk_row.extend([0])
                    if params.inc_explanation:
                        bulk_row.extend([''])

            bulk_data.append(bulk_row)

        return pd.DataFrame(bulk_data, columns = bulk_columns)