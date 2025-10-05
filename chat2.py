from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

# Load model
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=500
    )
)

st.header("Ayush Chat Bot")

# Create chat model
model = ChatHuggingFace(llm=llm)

# Input box
user_prompt = st.text_input("Enter your prompt:")

# Button
if st.button("summarize"):
    if user_prompt:   # only run if text is entered
        result = model.invoke(user_prompt)
        st.write(result.content)   # show result on screen
    else:
        st.warning("Please enter a prompt first!")
