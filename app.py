import gradio as gr
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

load_dotenv()  
MY_API_KEY = os.getenv("MY_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_video_and_ask(youtube_url, user_query):
    try:
        # Step 1: Corpus (Extraction)
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()

        # Step 2: Indexing (FAISS)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        vector_db = FAISS.from_documents(chunks, embeddings)

        # Step 3: Generation (Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=MY_API_KEY
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )

        result = qa_chain.invoke(user_query)
        return result["result"]

    except Exception as e:
        return f"Error: {str(e)}"

# --- UI ---
demo = gr.Interface(
    fn=process_video_and_ask,
    inputs=[gr.Textbox(label="YouTube Link"), gr.Textbox(label="Question")],
    outputs=gr.Markdown(label="Bot Answer", height=500),
    title="YouTube RAG Assistant"
)

demo.launch(debug=True)
