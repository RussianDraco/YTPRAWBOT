from better_profanity import profanity

def clean_title(raw_title):
    #return title.decode('ascii', 'ignore')
    title = swear_check(raw_title)
    output = ""
    for character in title:
        if character.isalnum() or character == " ":
            output += character
    return output

def swear_check(text):
    return profanity.censor(text, '*')