"""Bulk output"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903

import pandas as pd
from dataclasses import dataclass

from backend.base_classes import ScoreResultItem, TopicDefinition
from backend.gold_data import GoldData

@dataclass
class BulkOutputParams:
    """Parameters for create output data"""
    inc_explanation  : bool
    inc_source       : bool
    inc_gold_data    : bool
    site_map_only    : bool

class BulkOutput():
    """Class to build output data"""

    def calculate_gold_data(
            self,
            url : str,
            primary_topic   : str,
            secondary_topic : str,
            gold_data_dict : dict[str, GoldData],
            topic_dict     : dict[str, int],
            is_primary : bool
        ) -> []:
        """Return primary gold data"""
        if not gold_data_dict:
            return [None, None]
        u_url = url.lower().strip()
        if u_url not in gold_data_dict:
            return [None, None]
        
        gold_data = gold_data_dict[u_url]
        if is_primary:
            first_topic       = primary_topic
            second_topic      = secondary_topic
            first_gold_topic  = gold_data.primary_topic
            second_gold_topic = gold_data.secondary_topic
        else:
            first_topic       = secondary_topic
            second_topic      = primary_topic
            first_gold_topic = gold_data.secondary_topic
            second_gold_topic = gold_data.primary_topic

        if not first_gold_topic and is_primary:
            return [None, None]

        if not second_gold_topic and not is_primary:
            return [None, None]

        if first_gold_topic and first_gold_topic.lower().strip() not in topic_dict:
            return [first_gold_topic, "ERROR GOLDEN DATA"]
        
        topic_correct = 0.0
        if first_topic.lower().strip() == first_gold_topic.lower().strip(): # exact fit
            topic_correct = 1.0
        elif second_topic.lower().strip() == second_gold_topic.lower().strip(): # exact fit
            topic_correct = 1.0
        elif first_topic.lower().strip() == second_gold_topic.lower().strip():
            topic_correct = 0.5
        elif second_topic.lower().strip() == first_gold_topic.lower().strip():
            topic_correct = 0.5

        return [first_gold_topic, topic_correct]

    def create_data(
            self,
            topic_list  : list[TopicDefinition],
            bulk_result : list[ScoreResultItem], 
            gold_data   : GoldData,
            params : BulkOutputParams
        ) -> pd.DataFrame:
        """Create output data"""
        
        if params.site_map_only:
            return self.build_sitemap_output(bulk_result)

        gold_data_dict = {}
        if params.inc_gold_data and gold_data and gold_data.data:
            gold_data_dict = {gd.url.lower().strip() : gd for gd in gold_data.data}

        topic_dict = {t.name.lower().strip() : t.id for t in topic_list}

        bulk_columns = ['URL', 'Input length', 'Extracted length', 'Lang', 'Translated length']
        bulk_columns.extend(['Primary', 'Primary score'])
        if params.inc_gold_data:
            bulk_columns.extend(['Gold Primary', 'Primary correct'])
        if params.inc_explanation:
            bulk_columns.extend(['Primary explanation'])
        bulk_columns.extend(['Secondary', 'Secondary score'])
        if params.inc_gold_data:
            bulk_columns.extend(['Gold Secondary', 'Secondary correct'])
        if params.inc_explanation:
            bulk_columns.extend(['Secondary explanation'])
        if params.inc_source:
            bulk_columns.extend(['Source text'])
        for topic_item in topic_list:
            bulk_columns.extend([f'[{topic_item.id}]{topic_item.name}'])
            if params.inc_explanation:
                bulk_columns.extend([f'[{topic_item.id}]Explanation'])

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
            
            if params.inc_gold_data:
                bulk_row.extend(self.calculate_gold_data(
                    row.current_url,
                    row.main_topics.primary.topic,
                    row.main_topics.secondary.topic,
                    gold_data_dict, 
                    topic_dict, 
                    True
                ))

            if params.inc_explanation:
                bulk_row.extend([row.main_topics.primary.explanation])

            bulk_row.extend([row.main_topics.secondary.topic, row.main_topics.secondary.topic_score]) # secondary topic

            if params.inc_gold_data:
                bulk_row.extend(self.calculate_gold_data(
                    row.current_url, 
                    row.main_topics.primary.topic, 
                    row.main_topics.secondary.topic, 
                    gold_data_dict, 
                    topic_dict, 
                    False
                ))

            if params.inc_explanation:
                bulk_row.extend([row.main_topics.secondary.explanation])

            if params.inc_source:
                bulk_row.extend([row.full_translated_text])
            
            score_data = row.ordered_result_score
            for topic_item in topic_list:
                if topic_item.id in score_data:
                    topic_score = score_data[topic_item.id]
                    bulk_row.extend([topic_score[0]])
                    if params.inc_explanation:
                        bulk_row.extend([topic_score[1]])
                else:
                    bulk_row.extend([0])
                    if params.inc_explanation:
                        bulk_row.extend([''])

            bulk_data.append(bulk_row)

        return pd.DataFrame(bulk_data, columns = bulk_columns)
    
    def build_sitemap_output(
        self,
        bulk_result : list[ScoreResultItem]
    ) -> pd.DataFrame:
        """Only build sitemap"""
        bulk_columns = ['URL']
        bulk_data = [r.current_url for r in bulk_result]
        return pd.DataFrame(bulk_data, columns = bulk_columns)