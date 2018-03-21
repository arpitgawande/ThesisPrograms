
# coding: utf-8

# In[98]:

#long_sentence = 'the beautiful flower from the farmers market'
stop_words = ["is", "a", "can", "the"]
token_stream = ["The", "beautiful", "girl", "from", "the", "farmers", "market", ".", "I", "like",
"chewing", "gum", "."]
sugg = []
previous = []
for i, w in enumerate(token_stream):
    #Dont consider stop words and single string words
    if w.lower() not in stop_words and len(w) > 1:
        sugg.append(w)
        if len(previous) ==  0:
            previous = [w] #built previous string.
        else: 
            sub = [w]
            #built list of substring by reversing the previous string and 
            # conacting in reverse order buiding one substring after another
            for s in reversed(previous): 
                sub = [s] + sub
                sugg.append(sub)
            previous = sub
    #If stop word, then start substring after than 
    else:
        previous = []


# In[99]:

for s in sugg:
    print(s)


# In[82]:

'the' in ['the']


# In[ ]:



