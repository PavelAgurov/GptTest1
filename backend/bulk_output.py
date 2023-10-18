"""Bulk output"""

# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903,W1203

import pandas as pd
from dataclasses import dataclass

from backend.base_classes import ScoreResultItem, TopicDefinition
from backend.gold_data import GoldenData
from utils.utils import str2lower

@dataclass
class BulkOutputParams:
    """Parameters for create output data"""
    inc_explanation  : bool
    inc_source       : bool
    inc_golden_data    : bool
    site_map_only    : bool

class BulkOutput():
    """Class to build output data"""

    def get_golden_data_array(
            self,
            url : str,
            gold_data_dict    : dict[str, GoldenData],
            topic_dict        : dict[str, TopicDefinition],
            score_data        : dict[int, tuple[float, str]],
            is_primary        : bool
        ) -> []:
        """Return primary golden data"""
        # no golden data at all
        if not gold_data_dict:
            return [None, None, None, None]
        
        # no golden data for this URL
        u_url = url.lower().strip()
        if u_url not in gold_data_dict:
            return [None, None, None, None]
        
        golden_data = gold_data_dict[u_url]

        if is_primary:
            first_golden_topic_name = golden_data.primary_topic
        else:
            first_golden_topic_name = golden_data.secondary_topic

        if not first_golden_topic_name:
            return [None, None, None, None]
        
        if str2lower(first_golden_topic_name) not in topic_dict:
            return [first_golden_topic_name, "ERROR GOLDEN DATA", None, None]
        
        first_golden_topic = topic_dict[str2lower(first_golden_topic_name)]
        first_golden_topic_index = first_golden_topic.id
        first_golden_topic_priority = first_golden_topic.priority

        first_golden_score = 0
        first_topic_expl   = None
        if score_data:
            if first_golden_topic_index in score_data:
                first_golden_score = score_data[first_golden_topic_index][0]
                first_topic_expl   = score_data[first_golden_topic_index][1]
            else:
                first_golden_score = "ERROR"

        return [first_golden_topic_name, first_golden_score, first_topic_expl, first_golden_topic_priority]

    def get_main_score(
            self,
            url                  : str,
            gold_data_dict       : dict[str, GoldenData],
            main_topic_primary   : str,
            main_topic_secondary : str
    ) -> list[int]:
        """Get main score"""
        # no golden data at all
        if not gold_data_dict:
            return [1, 1, 1]
        
        # no golden data for this URL
        u_url = url.lower().strip()
        if u_url not in gold_data_dict:
            return [1, 1, 1]
        golden_data = gold_data_dict[u_url]

        golden_array = set([str2lower(golden_data.primary_topic, ''), str2lower(golden_data.secondary_topic, '')])
        main_array   = set([str2lower(main_topic_primary, '')       , str2lower(main_topic_secondary, '')       ])
        
        main_score = 0
        intersection_len = len(golden_array.intersection(main_array))
        if intersection_len == 2:
            main_score = 1
        elif intersection_len == 1:
            main_score = 0.5

        primary_score = 0
        if str2lower(main_topic_primary) in golden_array:
            primary_score = 1

        secondary_score = 0
        if str2lower(main_topic_secondary) in golden_array:
            secondary_score = 1

        return [main_score, primary_score, secondary_score]

    def create_data(
            self,
            topic_list  : list[TopicDefinition],
            bulk_result : list[ScoreResultItem], 
            golden_data   : GoldenData,
            params : BulkOutputParams
        ) -> pd.DataFrame:
        """Create output data"""
        
        if params.site_map_only:
            return self.build_sitemap_output(bulk_result)

        golden_data_dict = {}
        if params.inc_golden_data and golden_data and golden_data.data:
            golden_data_dict = {gd.url.lower().strip() : gd for gd in golden_data.data}

        topic_dict = {t.name.lower().strip() : t for t in topic_list}

        bulk_columns = ['URL', 'Input length', 'Extracted length', 'Lang', 'Translated length']
        if params.inc_golden_data:
            bulk_columns.extend(['Main correct', 'Primary correct', 'Secondary correct'])

        bulk_columns.extend(['URL detector', 'Leaders', 'Senior leaders count 1', 'Senior leaders count 2'])
        bulk_columns.extend(['Primary', 'Primary score'])
        if params.inc_golden_data:
            bulk_columns.extend([
                'Golden Primary', 
                'Golden primary score', 
                'Golden primary explanation', 
                'Golden primary priority'
            ])
        if params.inc_explanation:
            bulk_columns.extend(['Primary explanation'])
        bulk_columns.extend(['Secondary', 'Secondary score'])
        if params.inc_golden_data:
            bulk_columns.extend([
                'Golden Secondary', 
                'Golden secondary score', 
                'Golden secondary explanation', 
                'Golden secondary priority'
            ])
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
            if row.error:
                erorr_row = [row.current_url, row.error]
                erorr_row.extend([None]*(len(bulk_columns)-2))
                bulk_data.append(erorr_row)
                continue

            bulk_row = []
            bulk_row.extend([row.current_url, row.input_text_len, row.extracted_text_len, row.translated_lang, row.transpated_text_len])

            if params.inc_golden_data:
                bulk_row.extend(self.get_main_score(
                    row.current_url,
                    golden_data_dict,
                    row.get_main_topic_primary(),
                    row.get_main_topic_secondary(),
                ))

            bulk_row.extend([row.topics_by_url_info, row.leaders_list_str, row.senior_leaders_1, row.senior_leaders_2])

            if row.main_topics and row.main_topics.primary:
                bulk_row.extend([row.main_topics.primary.topic, row.main_topics.primary.topic_score]) # primary topic
            else:
                bulk_row.extend([None, None])
            
            score_data = row.ordered_result_score

            if params.inc_golden_data:
                bulk_row.extend(self.get_golden_data_array(
                    row.current_url,
                    golden_data_dict, 
                    topic_dict,
                    score_data,
                    True
                ))

            if params.inc_explanation:
                if row.main_topics and row.main_topics.primary:
                    bulk_row.extend([row.main_topics.primary.explanation])
                else:
                    bulk_row.extend([None])

            if row.main_topics and row.main_topics.secondary:
                bulk_row.extend([row.main_topics.secondary.topic, row.main_topics.secondary.topic_score]) # secondary topic
            else:
                bulk_row.extend([None, None])

            if params.inc_golden_data:
                bulk_row.extend(self.get_golden_data_array(
                    row.current_url,
                    golden_data_dict,
                    topic_dict,
                    score_data,
                    False
                ))

            if params.inc_explanation:
                if row.main_topics and row.main_topics.secondary:
                    bulk_row.extend([row.main_topics.secondary.explanation])
                else:
                    bulk_row.extend([None])

            if params.inc_source:
                bulk_row.extend([row.full_translated_text])
            
            for topic_item in topic_list:
                if score_data and topic_item.id in score_data:
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