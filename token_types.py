import re
import string
import unicodedata
from enum import Enum

class TokenType(Enum):
    SUFFIX = 1
    MAYBE_PREFIX = 2
    PUNCTUATION = 3
    INVALID = 4

def is_punctuation(token):
    return token in string.punctuation
	
def is_suffix(token):
    # A decoded token is a suffix if it's not punctuation and have no leading white space
    return not is_punctuation(token) and token[0] != ' '

def contains_special_char(token):
    # Define a regex pattern to match all non-alphanumeric and non-whitespace characters
    pattern = r'[^\w\s]'
    # Use re.findall() to find all matches in the token
    return  re.findall(pattern, token).count > 0

def contains_non_latin(token):
    return any(char in string.punctuation or not unicodedata.in1d('Latin', char) for char in token)

def is_invalid(token):
    if is_punctuation(token):
        return False
    else:
        # Any token that contains special or non-Latin character is invalid
        return contains_special_char(token) or contains_non_latin(token)

def get_token_type(token):
    if is_punctuation(token):
        return TokenType.PUNCTUATION
    elif is_suffix(token):
        return TokenType.SUFFIX
    elif is_invalid(token):
        return TokenType.INVALID
    else:
        return TokenType.MAYBE_PREFIX