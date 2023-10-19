"""
    Excluded sentenses
"""

def is_excluded_sentense(txt : str) -> bool:
    """Allowed text"""
    if txt.startswith('Find out more about') and txt.endswith('Click here'):
        return True
    if txt.startswith('Cookie Preferences'):
        return True
    if txt.startswith('Learn more about'):
        return True
    return False
