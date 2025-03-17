import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google AI and set up model and memory
genai.configure(api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

context = """
          You are an expert in Data Science and Machine Learning. You help students and
          professionals solve problems related to data analysis, machine learning algorithms, 
          statistical methods, and data visualization.
          """
conversation = ConversationChain(memory=memory, llm=llm, prompt=PromptTemplate(template=context))


# Streamlit UI
st.title("🤖 AI Data Science Tutor")

# Create a user input field
user_input = st.text_input("Ask a Data Science Question:")

# Initialize conversation state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if user_input:
    # Pass the input to LangChain for processing
    response = conversation.predict(input=user_input)

    # Update conversation history
    st.session_state["conversation_history"].append({"user": user_input, "ai": response})

# Display conversation history
if st.session_state["conversation_history"]:
    for interaction in st.session_state["conversation_history"]:
        st.write(f"**User**: {interaction['user']}")
        st.write(f"**AI Tutor**: {interaction['ai']}")
