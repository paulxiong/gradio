import sys
import os

sys.path.append(os.getcwd())
root_dir = os.getcwd()
sys.path.insert(1,root_dir) 
import gradio_my.gradio_my as gr

title = "Ask Rick a Question"
description = """
<center>
The bot was trained to answer questions based on Rick and Morty dialogues. Ask Rick anything!
<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
</center>
"""

article = "Check out [the original Rick and Morty Bot](https://huggingface.co/spaces/kingabzpro/Rick_and_Morty_Bot) that this demo is based off of."

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("ericzhou/DialoGPT-Medium-Rick_v2")
model = AutoModelForCausalLM.from_pretrained("ericzhou/DialoGPT-Medium-Rick_v2")

def predict(input):
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # generate a response 
    history = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()

    # convert the tokens to text, and then split the responses into the right format
    response = tokenizer.decode(history[0]).split("<|endoftext|>")
    return response[1]

gr.Interface(fn = predict, inputs = ["textbox"], outputs = ["text"], title = title, description = description, article = article).launch() 
