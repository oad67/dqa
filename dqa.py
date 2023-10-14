import streamlit as st
import numpy as np
import pandas as pd
import torch


from transformers import pipeline
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer

st.set_page_config(page_title="Contextual QnA App_OA")
st.write("QnA App_OA")

model_name = "deepset/roberta-base-squad2"

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

text=st.text_area("Enter text to summarize:")

clicked=st.button("Answer")


QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}

res = nlp(QA_input)




if st.button("Summarize"):
 st.write(res)

