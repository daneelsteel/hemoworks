def clear_text(text,trash_tokens):
    text=text.strip(trash_tokens)
    return text

def get_words(text):

    trash_tokens=',.)(!"£$%^&*()~#@#:;^&\\?/.¬—-~<>«»'
    tokens=text.split()
    good_tokens=[]
    for token in tokens:
        clean_token=clear_text(token,trash_tokens)
        if clean_token!='':
            clean_token=clean_token.lower()
            good_tokens.append(clean_token)
    return good_tokens

def main():
    filename='C:\\Users\\Эрнеста\\Desktop\\quotes.txt'
    DASH='—'
    list_authors=[]
    with open (filename,encoding='utf-8') as fid:
        for line in fid:
            parts=line.split(DASH)
            quote=parts[0].strip()
            author=parts[1].strip()

            quote_words=get_words(quote)
            
            if 'разум' in quote_words:
                list_authors.append(author)
    print(
        'your word has been found in ',
        len(list_authors))
    
    print(
        'quotes by: ',
        ','.join(list_authors))
                

if __name__=='__main__':
    main()
