from youtube_transcript_api import YouTubeTranscriptApi
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
import streamlit as st

import re

load_dotenv()

key=os.getenv("API_KEY")

def store_transcript_string(video_id):
    ytt_api=YouTubeTranscriptApi()
    try:
        transcript=ytt_api.fetch(video_id,languages=['en'])
        
    except:
        print("No english transcript found and its only designed to get you the english transcript")
        return
    
    file_name=input("name the file name(without extension):")
    raw_data=transcript.to_raw_data()
    string_transcript=" ".join([ data['text'] for data in raw_data])
    with open(f"{file_name}.txt","w",encoding="utf-8") as f:
        f.write(string_transcript)
    print(f"Transcript saved to {file_name}.txt")

def generate_transcript(video_id):
    ytt_api=YouTubeTranscriptApi()
    try:
        transcript=ytt_api.fetch(video_id,languages=['en'])  
    except:
        print("No english transcript found and its only designed to get you the english transcript")
        return
    
    raw_data=transcript.to_raw_data()    
    string_transcript=" ".join([ data['text'] for data in raw_data])
    return string_transcript

def split_text(video_id):
    
    splitter=RecursiveCharacterTextSplitter(
    chunk_size=202,
    chunk_overlap=0
    )

    split_text=splitter.split_text(generate_transcript(video_id))
    return [Document(page_content=chunk) for chunk in split_text]


def build_vectorstore(video_id):

    embedding_model = OpenAIEmbeddings(
    base_url="https://aipipe.org/openai/v1",
    api_key=key,
    model='text-embedding-3-large'
    )

    docs = split_text(video_id)  # List[Document]
    vectorstore = FAISS.from_documents(documents=docs, embedding=embedding_model)
    return vectorstore



def retrieve_context(video_id, query):
    vectorstore = build_vectorstore(video_id)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )

    results = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in results])
    return context


def generate_prompt(context,question):

    prompt=PromptTemplate(
        template="""
        You are a  teaching assistant.

        Given the following transcript context from a YouTube video, answer the user's question in a clear, detailed, and structured manner.

        Always use full sentences, provide explanations, and refer to concepts from the transcript. If the context is insufficient, just say "I don't know".

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )
    formatted_prompt = prompt.format(context=context, question=question)
    return formatted_prompt


def get_response(video_id,query):

    llm=ChatOpenAI(
    base_url="https://aipipe.org/openrouter/v1",
    model='openai/gpt-4.1',
    api_key=key
    )
    
    
    context = retrieve_context(video_id, query)

    response=llm.invoke(generate_prompt(context,query))

    return response.content

def extract_video_id(url):
    # This pattern matches most YouTube video URL formats
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


#https://www.youtube.com/watch?v=LPZh9BOjkQs
# Streamlit UI
st.set_page_config(page_title="YouTube Chatbot", page_icon="🎥", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF6347;'>🎥 YouTube Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions about any YouTube video by pasting the URL below and entering your query.</p>", unsafe_allow_html=True)

# Input fields
with st.container():
    st.subheader("🔗 Enter YouTube Video URL:")
    video_url = st.text_input("YouTube link:", placeholder="e.g. https://www.youtube.com/watch?v=LPZh9BOjkQs")

    st.subheader("❓ Ask a Question:")
    query = st.text_input("What do you want to know from the video?", placeholder="e.g. What is this video about?")

# Process on button click
if st.button("🧠 Get Answer"):
    if not video_url or not query:
        st.warning("Please enter both the video URL and your question.")
    else:
        with st.spinner("Processing..."):
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please try again.")
            else:
                try:
                    answer = get_response(video_id, query)
                    st.success("✅ Here's the answer:")
                    st.markdown(f"**{answer}**")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
