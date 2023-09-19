"""
    Gold data methods
"""
# pylint: disable=C0301,C0103,C0303,C0304,C0305,C0411,E1121,R0903

import os
from dataclasses import dataclass
import pandas as pd

@dataclass
class GoldDataItem:
    """Gold data item"""
    url             : str
    primary_topic   : str
    secondary_topic : str

@dataclass
class GoldData:
    """Gold data"""
    data  : list[GoldDataItem]
    error : str


GOLDEN_DATA_FILE = r'golden-data\\golden_data.xlsx'

def get_gold_data() -> GoldData:
    """Load gold data"""
    
    if not os.path.isfile(GOLDEN_DATA_FILE):
        return None

    try:
        gold_data_excel = pd.read_excel(GOLDEN_DATA_FILE)
        gold_data_excel = gold_data_excel.fillna('')  
    except Exception as error: # pylint: disable=W0718
        print(error)
        return GoldData(None, error)

    if gold_data_excel.shape[1] < 3:
        print('Golden data must have 3 columns: URL, primary topic, secondary topic')
        return None

    def strip_str(s):
        if s is None:
            return ''
        s = str(s)
        return s.strip()

    gold_data = list[GoldDataItem]()
    for row in gold_data_excel.values:
        url = strip_str(row[0])
        if not url: # allow empty lines
            continue
        if url.startswith('#'): # allow comments in the file
            continue
        primary   = strip_str(row[1]) # line without data
        secondary = strip_str(row[2])
        if not primary and not secondary:
            continue

        gold_data.append(GoldDataItem(
            url,
            primary,
            secondary
        ))

    error = None
    url_set = [d.url for d in gold_data]
    if len(url_set) != len(list(set(url_set))):
        error = "golden_data.xlsx has duplicated URLs"

    return GoldData(gold_data, error)
