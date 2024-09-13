import os
from constants import google_key
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
import streamlit as st

# Initialize Google Gemini API key
os.environ["GOOGLE_API_KEY"] = google_key

# Initialize Streamlit framework
st.title("CelebInsight")
input_text = st.text_input("Search for a celebrity:")

# Prompt Templates
biography_prompt = PromptTemplate(
    input_variables=["name"],
    template="Provide a brief biography of {name} in one sentence.",
)

dob_prompt = PromptTemplate(
    input_variables=["name"],
    template="What is the date of birth of {name}?",
)

recent_award_prompt = PromptTemplate(
    input_variables=["name"],
    template="What is the most recent major award or recognition received by {name}?",
)

recent_controversy_prompt = PromptTemplate(
    input_variables=["name"],
    template="What is the most recent controversy or challenge faced by {name}?",
)

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# Memory Initialization
biography_memory = ConversationBufferMemory(
    input_key="name", memory_key="biography_history"
)
dob_memory = ConversationBufferMemory(input_key="name", memory_key="dob_history")
award_memory = ConversationBufferMemory(input_key="name", memory_key="award_history")
controversy_memory = ConversationBufferMemory(
    input_key="name", memory_key="controversy_history"
)

# Chain 1: Fetch celebrity biography
biography_chain = LLMChain(
    llm=llm,
    prompt=biography_prompt,
    verbose=True,
    output_key="biography",
    memory=biography_memory,
)

# Chain 2: Fetch celebrity's date of birth
dob_chain = LLMChain(
    llm=llm,
    prompt=dob_prompt,
    verbose=True,
    output_key="dob",
    memory=dob_memory,
)

# Chain 3: Fetch recent major award
award_chain = LLMChain(
    llm=llm,
    prompt=recent_award_prompt,
    verbose=True,
    output_key="award",
    memory=award_memory,
)

# Chain 4: Fetch recent controversy
controversy_chain = LLMChain(
    llm=llm,
    prompt=recent_controversy_prompt,
    verbose=True,
    output_key="controversy",
    memory=controversy_memory,
)

# Combine chains into a SequentialChain
parent_chain = SequentialChain(
    chains=[biography_chain, dob_chain, award_chain, controversy_chain],
    verbose=True,
    input_variables=["name"],
    output_variables=["biography", "dob", "award", "controversy"],
)

if input_text:
    result = parent_chain({"name": input_text})
    st.write(result)

    with st.expander("Biography"):
        st.info(result.get("biography", "No information available"))
    with st.expander("Date of Birth"):
        st.info(result.get("dob", "No information available"))
    with st.expander("Recent Award"):
        st.info(result.get("award", "No information available"))
    with st.expander("Recent Controversy"):
        st.info(result.get("controversy", "No information available"))
