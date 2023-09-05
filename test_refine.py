"""
    Tests
    To run: pytest
"""

# pylint: disable=C0103,R0915,C0301,C0411

import pytest

from backend.llm.refine import RefineChain

def simple_len_function(text : str):
    """Simple len function only for test"""
    return len(text)

def test_refine():
    """Test of refine"""
    refine = RefineChain(None)
    sentence_list = [
        'Sentence number 001',
        'Sentence number 002',
        'Sentence number 003',
        'Sentence number 004',
        'Sentence number 005',
        'Sentence number 006',
        'Sentence number 007'
    ]
    current_index = 0
    new_index = refine.get_max_possible_index(sentence_list, current_index, 10, simple_len_function)
    assert new_index == 0
    current_index = new_index+1
    new_index = refine.get_max_possible_index(sentence_list, current_index, 10, simple_len_function)
    assert new_index == 1

    part_count = 0
    current_index = 0
    for _ in range(100):
        new_index = refine.get_max_possible_index(sentence_list, current_index, 20, simple_len_function)
        current_index = new_index+1
        part_count+=1
        if new_index >= len(sentence_list):
            break
    assert part_count == 4

    part_count = 0
    current_index = 0
    for _ in range(100):
        new_index = refine.get_max_possible_index(sentence_list, current_index, 1000, simple_len_function)
        current_index = new_index+1
        part_count+=1
        if new_index >= len(sentence_list):
            break
    assert part_count == 1
