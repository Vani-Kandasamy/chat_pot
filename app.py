from openai import OpenAI
import streamlit as st
from rag import QA_MODEL, streaming_question_answering, get_similar_context

FIRST_MESSAGE = {"role": "assistant", "content": "Hello there!!!!, I am a highly trained child supportive bot. You can ask me any queries about the child care."}
IMAGE_ADDRESS = "https://upworthyscience.com/media-library/a-boy-playing-with-ipal-a-social-robot.jpg?id=24321363&width=2000&height=1500&quality=85&coordinates=200%2C0%2C200%2C0"

# set the title
st.title("Child Supporter 👩‍🍼")
# set the image
st.image(IMAGE_ADDRESS, caption = 'Child Supporter')

st.subheader("Chat with Us 🤖")
# set openai key
client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = QA_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = [FIRST_MESSAGE]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        pinecone_context = get_similar_context(prompt)
        response = st.write_stream(streaming_question_answering(prompt, pinecone_context))
    st.session_state.messages.append({"role": "assistant", "content": response})