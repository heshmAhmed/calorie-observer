from app.PreProcessing import preprocess
import re
import json
import nltk
import difflib
import enchant
from nltk.stem import WordNetLemmatizer
import pprint
nltk.download('wordnet')

dict = {}
allwords = []


def loadFromDisk(path):
    data = None
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def removeTrash(text):
    regex = re.compile("[^a-zA-Z0-9 ' ' \n . %]")
    return regex.sub(" ", text)


# seperate digists by space
def separateDigits(text):
    str1 = ""
    for i in range(len(text)):
        if re.match("^[0-9]+[.][0-9]+|[0-9]+$", text[i]) or text[i] == "%":
            if text[i - 1].isalpha():
                str1 += " "
            if text[i + 1].isalpha() or text[i + 1] == "%":
                str1 += text[i] + " "
                continue
        str1 += text[i]
    return str1



# correct only to our dictionary not to all english words
def correctText(text):
    lemmatizer = WordNetLemmatizer()
    checker = enchant.Dict("en_US")
    list_str = text.split()
    for i in range(len(list_str)):
        token = list_str[i]
        if not checker.check(token):
            result = difflib.get_close_matches(token, allwords, 1)
            token = result[0] if type(result) == list and len(result) > 0 else token
        else:
            token = lemmatizer.lemmatize(token)
        list_str[i] = token 
    return list_str


def isNum(chr):
    return re.match("^[0-9]+[.][0-9]+|[0-9]+$", chr)


def isValid(chr):
    return chr == "g" or chr == "m" or chr == "mg" or chr == "%"



def tokenize(list_str):
    global dict
    ls = []
    i = 0
    while i < len(list_str):
        if i < len(list_str) - 1 and list_str[i] + " " + list_str[i + 1] in dict:
            ls.append(list_str[i] + " " + list_str[i + 1])
            i += 1
        elif list_str[i] in dict:
            ls.append(list_str[i])
        elif isNum(list_str[i]):
            ls.append(list_str[i])
        elif i > 0 and isValid(list_str[i]) and (len(ls) > 0 and isNum(ls[len(ls) - 1])):
            ls.append(list_str[i])
        i += 1
    return ls


def match(list_str):
    global dict
    i = 0
    while i < len(list_str):
        key = list_str[i]
        value = []
        if key in dict:
            if dict[key] is not None:
                value = dict[key]
            i += 1
            while i < len(list_str) and list_str[i] not in dict:
                value.append(list_str[i])
                i += 1
            dict[key] = value
        else:
            i += 1


def postprocess(img):
    global dict, allwords
    dict = loadFromDisk("app/data/temp.json")
    allwords = loadFromDisk("app/data/allwords.json")
    text = preprocess(img)

    # text = (
    #     "nutrition facts \n "
    #     + "saturated 3.1g 139%  11.5g \n"
    #     + "per 1 sandwich 135 ml"
    #     + "amount % dally value \n"
    #     + "calories 260 \n"
    #     + "fat8g 12   % \n"
    #     + "ur. \n"
    #     + "ames 16 % \n"
    #     + "chlesterol 5 mg \n"
    #     + "sodium 240 mg 10% \n"
    #     + "total carbohydrate 43 g 14%\n "
    #     + "fibre 1g 4a% \n"
    #     + "sugars 23g \n"
    #     + "protein4g \n"
    #     + "vitamin a oo% vitamin c o%m \n"
    #     + "catciurm 6% lron 15% "
    # )

    print("Tesseract output--------------------------------------")
    print(text)

    print()
    print("Some preprocessing------------------------------------------------------")
    text = removeTrash(text)
    text = text.lower()
    text = separateDigits(text)
    print(text)

    print()
    print("Correct text output-----------------------------------")
    list_str = correctText(text)
    print(list_str)

    print()
    print("Tokenize---------------------------------------")
    list_str = tokenize(list_str)
    print(list_str)

    print()
    print("Match output------------------------------------------")
    match(list_str)
    pprint.pprint(dict)
    return dict
