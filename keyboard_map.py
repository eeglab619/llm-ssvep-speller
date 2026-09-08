CLASS_TO_CHAR = {}
for i in range(26):
    CLASS_TO_CHAR[i] = chr(ord('A') + i)
for i in range(10):
    CLASS_TO_CHAR[26 + i] = str(i)
CLASS_TO_CHAR[36] = ' '
CLASS_TO_CHAR[37] = ','
CLASS_TO_CHAR[38] = '.'
CLASS_TO_CHAR[39] = "'"

CHAR_TO_CLASS = {v: k for k, v in CLASS_TO_CHAR.items()}
ALLOWED_CHARS = set(CHAR_TO_CLASS.keys())
