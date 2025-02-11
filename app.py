from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr
import os
from google.cloud import speech_v1p1beta1 as speech
import io
from googletrans import Translator

# Function to extract the YouTube transcript
def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        # If no transcript exists, return None for fallback processing
        return None

# Extract audio and convert to text (fallback for videos without transcript)
def extract_audio_and_convert_to_text(video_url):
    video_id = video_url.split("v=")[1]
    video_stream_url = f"https://www.youtube.com/watch?v={video_id}"

    # Use youtube-dl or other methods to download audio (skipped in this example)
    # For simplicity, here we assume you've downloaded the audio as 'audio.wav'
    audio_file = "audio.wav"

    # Use Google Cloud Speech-to-Text for transcription
    client = speech.SpeechClient()
    with io.open(audio_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",  # Can be set to Hindi or other languages
    )

    response = client.recognize(config=config, audio=audio)
    transcript_text = " ".join([result.alternatives[0].transcript for result in response.results])
    return transcript_text

# Translate text to English if it's not already in English
def translate_to_english(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Check language and translate if needed
def process_transcript_for_language(transcript, language_code='en'):
    if language_code != 'en':
        transcript = translate_to_english(transcript, target_language='en')
    return transcript

# Function to store the transcript into FAISS
def store_transcript(transcript):
    # Split transcript into chunks
    doc_chunks = [Document(page_content=sentence) for sentence in transcript.split('. ') if sentence]

    # Use HuggingFace embeddings and FAISS for efficient storage
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(doc_chunks, embeddings)
    return db

# Function to retrieve information from the FAISS index
def retrieve_information(db, query):
    # Retrieve relevant context for the user query
    docs = db.similarity_search(query)
    content = "\n".join([x.page_content for x in docs])
    return content

# Function to generate the response using LLM (ChatGoogleGenerativeAI)
def generate_response(query, context):
    qa_prompt = """
    Use the following pieces of context to answer the user's question.
    ----------------
    """
    input_text = qa_prompt + "\nContext:\n" + context + "\nUser question: " + query
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    result = llm.invoke(input_text)
    return result.content

# Function to handle the Gradio interface and workflow
def process_data(youtube_url, query):
    transcript = get_youtube_transcript(youtube_url)

    # If transcript is not available, extract audio and convert to text
    if not transcript:
        transcript = extract_audio_and_convert_to_text(youtube_url)

    # If transcript still not available, return error
    if not transcript:
        return "Error: Could not retrieve transcript or audio."

    # Process the transcript for language translation if needed
    transcript = process_transcript_for_language(transcript)

    # Store the transcript in FAISS
    db = store_transcript(transcript)

    # Retrieve context and generate response
    retrieved_content = retrieve_information(db, query)
    response = generate_response(query, retrieved_content)

    return response

# Set your Google API key for authentication
os.environ["GOOGLE_API_KEY"] = "AIzaSyCBHUxuIrv2VvkDWoOmcKd-qoHB2udu1HA"

# Create the Gradio interface
iface = gr.Interface(
    fn=process_data,
    inputs=[gr.Textbox(lines=1, placeholder="Enter YouTube video URL"), gr.Textbox(lines=1, placeholder="Enter your query")],
    outputs="text",
    title="YouTube Video Q&A",
    description="Enter a YouTube video URL and ask a question about it.",
)

# Launch the interface
iface.launch()
