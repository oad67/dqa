import streamlit as st
import numpy as np
import pandas as pd
import torch


from transformers import pipeline
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer

model_name = "deepset/roberta-base-squad2"

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

text=st.text_area("Enter text to summarize:")

clicked=st.button("Answer")


QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}

res = nlp(QA_input)


st.set_page_config(page_title="Contextual QnA App_OA")
st.write("QnA App_OA")

if st.button("Summarize"):
 st.write(res)

#import nltk

#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize, sent_tokenize


#nltk.download("stopwords")
#nltk.download("punkt")

#import joblib

#st.set_page_config(page_title="Text Summarizer_OA")
#st.write("Text Summarizer")

#text=st.text_area("Enter text to summarize:")

#clicked=st.button("Summarize")
####

#if st.button("Summarize"):
 #   import pandas as pd
#    import numpy as np


#    from nltk.corpus import stopwords
#    from nltk.tokenize import word_tokenize, sent_tokenize

#    sw=set(stopwords.words('english'))
#    words=word_tokenize(text)

#    freqTable=dict()
 #   for word in words:
  #      word=word.lower()
   #     if word in sw:
    #        continue
     #   if word in freqTable:
      #      freqTable[word]+=1
       # else: freqTable[word]=1
    
    #sentences=sent_tokenize(text)
   # sentencevalue=get_sentencevalue()

    #def get_sumvalues():
     #   sumvalues=0
      #  for sentence in sentencevalue:
       #     sumvalues+=sentencevalue[sentence]
        
        #average=int(sumvalues/len(sentencevalue))
        #return(average)

 #   average=get_sumvalues()

#    summary=''
#    for sentence in sentences:
#        if(sentence in sentencevalue) and (sentencevalue[sentence]>(1.2*average)):
#            summary+=" "+sentence

 #   st.write(summary)
