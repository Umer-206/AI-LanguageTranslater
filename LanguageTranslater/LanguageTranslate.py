import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{input}"),
])

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit config
st.set_page_config(page_title="🌐 Langchain Translator", page_icon="🌍")

# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# UI Header
st.title("🌐 Langchain Translator using Gemini")
st.caption("Translate text into multiple languages and keep track of your recent translations.")

# User input
input_text = st.text_input("✍️ Enter text in any language:")

# Language selection
languages = [
    "Urdu", "German", "French", "Spanish", "Arabic",
    "Hindi", "Chinese", "Russian", "Turkish", "Japanese"
]
selected_language = st.selectbox("🌍 Select language to translate to:", languages)

# Translate button
if st.button("🔄 Translate") and input_text and selected_language:
    with st.spinner("Translating..."):
        response = chain.invoke({
            "input_language": selected_language,
            "output_language": selected_language,
            "input": input_text
        })

        # Save to history
        st.session_state.history.append({
            "original": input_text,
            "translated": response,
            "language": selected_language
        })

        # Show result
        st.markdown("### ✅ Translated Text:")
        st.success(response)

# Show translation history
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🕓 Translation History (Last 5)")
    for item in reversed(st.session_state.history[-5:]):
        with st.expander(f"➡️ {item['original']} → {item['language']}"):
            st.write(f"**Input:** {item['original']}")
            st.write(f"**Translated ({item['language']}):** {item['translated']}")

# Clear history button
if st.button("🗑️ Clear History"):
    st.session_state.history = []
    st.success("Translation history cleared.")
