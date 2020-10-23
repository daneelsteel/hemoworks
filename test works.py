#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from nltk import word_tokenize
from collections import Counter
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
from sklearn.metrics import accuracy_score

ua = UserAgent(verify_ssl=False)
session = requests.session()


# In[18]:


news = []
page_number = 0
for i in tqdm(range(4)):
    page_number += 1
    url = f'https://www.mk.ru/incident/{page_number}/'
    req = session.get(url, headers={'User-Agent': ua.random})
    print(req)
    req.encoding = 'utf-8'
    page = req.text
    soup = BeautifulSoup(page, 'html.parser')
    each_news = soup.find_all('div', {'class': 'listing-item__content'})
    if each_news:
        print(len(each_news))
    for one_piece in each_news:
        link = one_piece.find('a', {'class': 'listing-preview__content'}).attrs['href']
        news.append(link)


# In[19]:


print(len(news))


# In[81]:


link = news[10]
req = session.get(link, headers={'User-Agent': ua.random})
req.encoding = 'utf-8'
req = req.text
req = BeautifulSoup(req, 'html.parser')
new_text = req.find('div', {'class': "article__body"}) #.text.lower()
#print(new_text)
new_text = new_text.find_all('p') #.text.lower()
print(new_text)


# In[21]:


def get_full_news(link_list):
    texts = []
    titles = []
    for link in link_list:
        words = []
        req = session.get(link, headers={'User-Agent': ua.random})
        req.encoding = 'utf-8'
        req = req.text
        req = BeautifulSoup(req, 'html.parser')
        title = req.find('h1', {'class': "article__title"})
        titles.append(title)
        new_text = req.find('div', {'class': "article__body"})        
        if new_text:
            new_text = new_text.text.lower()
            for word in new_text:
                if word.isalpha():
                    
            texts.append(words)
    return titles, texts


# In[22]:


titles, texts = get_full_news(link_list)
if len(texts)==len(titles):
    print('all clear!')


# In[ ]:


df = pd.DataFrame(list(zip(titles, texts)), 
               columns =['titles', 'texts']) 
df.head()


# In[ ]:


df.to_csv('textbase.csv')


# In[96]:


word_form_instance = {}
parse_inst = morph.parse('синхрофазатроны')[0]
word_form_instance['word'] = parse_inst.word
word_form_instance['lemma'] = parse_inst.normal_form
word_form_instance['form'] = parse_inst.tag
word_form_instance['POS'] = parse_inst.tag.POS


# In[450]:


toy_df = pd.DataFrame({
   'EmployeeId': ['001', '002', '003', '004'],
   'City': ['я хорошая и дружелюбная сорока , меня хвалят. ', 'бегает. прыгает. ', 'смешной , и. расплывчатый', 'кто ты. кто я. '] 
})
toy_df


# In[451]:


import numpy as np

lst_col = 'City' 

x = toy_df.assign(**{lst_col:toy_df[lst_col].str.split('.')})

my_dataframe = pd.DataFrame({col:np.repeat(x[col].values, x[lst_col].str.len()) for col in x.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(x[lst_col].values)})[x.columns.tolist()]


# In[452]:


my_dataframe


# In[453]:


my_dataframe['words'] = my_dataframe['City']
lst_col = 'words' 

x = my_dataframe.assign(**{lst_col:my_dataframe[lst_col].str.split(' ')})

my_final_dataframe = pd.DataFrame({col:np.repeat(x[col].values, x[lst_col].str.len()) for col in x.columns.difference([lst_col])}).assign(**{lst_col:np.concatenate(x[lst_col].values)})[x.columns.tolist()]


# In[454]:


my_final_dataframe


# In[455]:


my_final_dataframe['lemma'] = my_final_dataframe.apply(lambda row: morph.parse(row.words)[0].normal_form, axis = 1)
my_final_dataframe['form'] = my_final_dataframe.apply(lambda row: morph.parse(row.words)[0].tag, axis = 1)
my_final_dataframe['POS'] = my_final_dataframe.apply(lambda row: morph.parse(row.words)[0].tag.POS, axis = 1)


# In[456]:


my_final_dataframe


# In[457]:


my_final_dataframe.drop(my_final_dataframe[my_final_dataframe.words == ''].index, inplace=True)


# In[458]:


my_final_dataframe


# In[467]:


punkt = [',', ';', ':', '"']
my_final_dataframe.drop(my_final_dataframe[my_final_dataframe.words.isin (punkt)].index, inplace=True)


# In[468]:


my_final_dataframe


# In[446]:


def get_morphology(word):
    word1 = word.strip('"')
    df_word = my_final_dataframe.loc[my_final_dataframe['words'] == word1]
    list_cities = []
    for index, row in df_word.iterrows():
        answer_word = str('This word is in:'+ row['City'] + ', which comes from the article:' + row['EmployeeId'])
        list_cities.append(answer_word)
    return list_cities
    
def get_tagged(word, allowed_tags=['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']):
    compl_word = word.split('+')
    tag = set(allowed_tags)&set(compl_word)
    if not tag:
        return ['this POS tag is not in my vocabulary. Sorry!']
        
    else:
        current_word = morph.parse(compl_word[0])[0]
        current_word_lemma = current_word.normal_form
        current_word_POS = current_word.tag.POS
        df_word_POS = my_final_dataframe.loc[(my_final_dataframe['lemma'] == current_word_lemma) & (my_final_dataframe['POS'] == current_word_POS)]
        list_taggers = []
        for index, row in df_word_POS.iterrows():
            answer_tagger = str('This word is in:'+ row['City'] + ', which comes from the article:' + row['EmployeeId'])
            list_taggers.append(answer_tagger)
        return list_taggers        
    
def get_lemma(word):
    word_lem_parsed = morph.parse(word)[0]
    word_lem_lemma = word_lem_parsed.normal_form
    df_lemma = my_final_dataframe.loc[(my_final_dataframe['lemma'] == word_lem_lemma)]
    list_lemmas = []
    for index, row in df_lemma.iterrows():
        answer_lemma = str('This word is in:'+ row['City'] + ', which comes from the article:' + row['EmployeeId'])
        list_lemmas.append(answer_lemma)
    return list_lemmas    
    
def get_pos(word):
    df_pos = my_final_dataframe.loc[(my_final_dataframe['POS'] == word)]
    list_poses = []
    for index, row in df_pos.iterrows():
        answer_pos = str('This word is in:'+ row['City'] + ', which comes from the article:' + row['EmployeeId'])
        list_poses.append(answer_pos)
    return list_poses
    
def work_with_words(parsed_quiery, allowed_tags=['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']):
    #inputs list, outputs list
    for word in parsed_quiery:
        trick = word[0]
        if trick=='"':
            print('word form')
            for i in get_morphology(word):
                print(i)
        elif '+' in word:
            print('lemma+POS')
            for i in get_tagged(word):
                print(i)
        elif word in allowed_tags:
            print('pos')
            for i in get_pos(word):
                print(i)
        else:
            print('lemma')
            for i in get_lemma(word):
                print(i)
    
def work_with_quiery(quiery):
    #inputs input string, outputs dataframe
    quiery_parsed = quiery.split(' ')
    if len(quiery_parsed)>2 or len(quiery_parsed)==0:
        print('Oh, come on, we talked about it!')
    else:
        work_with_words(quiery_parsed, allowed_tags=['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ'])


# In[447]:


allowed_tags=['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']
def main():
    quiery = input('What will we search for? ')
    answer = work_with_quiery(quiery)
    answer


# In[449]:


main()


# In[470]:


listing = ['fff']
if len(listing)==1:
    print('h')


# In[ ]:


allowed_tags = ['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']

def get_morphology(dataframe, word):
    word1 = word.strip('"')
    df_word = dataframe.loc[dataframe['words'] == word1]
    return df_word
    
def get_tagged(dataframe, word, allowed_tags=allowed_tags):
    compl_word = word.split('+')
    tag = set(allowed_tags)&set(compl_word)
    if not tag:
        return ['this POS tag is not in my vocabulary. Sorry!']
        
    else:
        current_word = morph.parse(compl_word[0])[0]
        current_word_lemma = current_word.normal_form
        current_word_POS = current_word.tag.POS
        df_word_POS = dataframe.loc[(dataframe['lemma'] == current_word_lemma) & (dataframe['POS'] == current_word_POS)]
        return df_word_POS
    
def get_lemma(dataframe, word):
    word_lem_parsed = morph.parse(word)[0]
    word_lem_lemma = word_lem_parsed.normal_form
    df_lemma = dataframe.loc[(dataframe['lemma'] == word_lem_lemma)]
    return df_lemma
    
def get_pos(dataframe, word):
    df_pos = dataframe.loc[(dataframe['POS'] == word)]
    return df_pos
    
def work_with_words(dataframe, word, allowed_tags=allowed_tags):
    trick = word[0]
    if trick=='"':
        print('word form')
        return get_morphology(dataframe, word)
    elif '+' in word:
        print('lemma+POS')
        return get_tagged(dataframe, word, allowed_tags=allowed_tags)
    elif word in allowed_tags:
        print('pos')
        return get_pos(dataframe, word)
    else:
        print('lemma')
        return get_lemma(dataframe, word)
    
def work_with_quiery(quiery):
    #inputs input string, outputs dataframe
    quiery_parsed = quiery.split(' ')
    if len(quiery_parsed)==1:
        answer = work_with_words(my_final_dataframe, quiery_parsed[0], allowed_tags=allowed_tags)
        list_words = []
        for index, row in answer.iterrows():
            answer_word = str('This word is in:'+ row['full_texts'] + ', which comes from the article:' + row['titles'])
            list_words.append(answer_word)
        return list_words
    elif len(quiery_parsed)==2:
        first_answer = work_with_words(my_final_dataframe, quiery_parsed[0], allowed_tags=allowed_tags)
        indexes = first_answer['ind'].tolist()
        indexes[:] = [x + 1 for x in indexes]
        
        second_answer = work_with_words(first_answer, quiery_parsed[1], allowed_tags=allowed_tags)
        


# ## MAIN

# NOUN	имя существительное
# 
# ADJF	имя прилагательное (полное)
# 
# ADJS	имя прилагательное (краткое)
# 
# COMP	компаратив
# 
# VERB	глагол (личная форма)
# 
# INFN	глагол (инфинитив)
# 
# PRTF	причастие (полное)
# 
# PRTS	причастие (краткое)
# 
# GRND	деепричастие
# 
# NUMR	числительное
# 
# ADVB	наречие
# 
# NPRO	местоимение-существительное
# 
# PRED	предикатив
# 
# PREP	предлог
# 
# CONJ	союз
# 
# PRCL	частица
# 
# INTJ	междометие
