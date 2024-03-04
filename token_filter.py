# We don't use string.punctuation since this list contains special char such as ">" or "*"
punctuations = ['"', '!', '\'', '(', ')', ',', '-', '.', ':', ';', '?']

def is_valid_punctuation(token):
    return token in punctuations

def is_basic_latin(char):
    return 0x0000 <= ord(char) <= 0x007F

def is_accent_latin(char):
    return 0x00C0 <= ord(char) <= 0x00D6 or 0x00D8 <= ord(char) <= 0x00F6

def is_latin(char):
    return is_basic_latin(char) or is_accent_latin(char)

def contains_all_latin(token):
    return all(is_latin(char) for char in token)

def is_valid_token(token):
    return is_valid_punctuation(token) or contains_all_latin(token)