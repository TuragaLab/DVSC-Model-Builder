import re

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def filename_strip(filename):
    keepcharacters = (' ', '.', ',', '_', '-', '(', ')', '\'', '\"', '+')
    return ''.join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    if(isinstance(text,tuple)):
        return [ atoi(c) for c in re.split('(\d+)', text[0]) ]
    else:
        return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_alphabetic_code(number):
    alpha_set = ['A','B','C','D','E','F','G', 'H', 'I', 'J']
    text = str(number)
    text = text.replace('.', 'X')
    text = text.replace(',', 'X')
    text = text.replace('-', 'Y')
    for i in range(0, 10):
        text = text.replace(str(i), alpha_set[i])
    return text