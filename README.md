# YouTube RAG Chat Assistant

An AI-powered chatbot that allows users to have a conversation with any YouTube video.
By using **Retrieval-Augmented Generation (RAG)**, the app extracts video transcripts, indexes them into a vector database, and uses Google's Gemini LLM to provide grounded, contextual answers.

## Features
- **Automated Transcript Extraction:** Instantly retrieves text data from any YouTube video with available English subtitles.
- **Efficient Semantic Search:** Uses FAISS to pinpoint exact video segments relevant to the user's query
- **Context-Grounded Answers:** Generates responses strictly based on the video's content
- **Output:** Displays responses in a clean, formatted style for better readability of complex explanations.

## Tech Stack
- **Frontend:** Gradio
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **LLM:** Google Gemini 2.5 Flash
- **LLM Framework:** LangChain
- **Vector Database:** FAISS (Facebook AI Similarity Search)


## Working
- **Extraction:** The app uses `YoutubeLoader` to pull the transcript text directly from the video.
- **Chunking:** The transcript is split into 1000-character pieces using `RecursiveCharacterTextSplitter` (to ensure semantic meaning is preserved).
- **Embedding:** Each chunk is converted into a 384-dimensional mathematical vector (embedding) using a Hugging Face model to capture semantic meaning.
- **Retrieval:** Based on the user's query, the app finds the top 4 (default) most relevant chunks from the video using the FAISS vector database.
- **Generation:** Gemini reads these specific chunks and the original question to generate a precise, grounded answer.

## How to Run
- Clone the Repo: git clone <link>
- Environment: Create a .env file and add MY_API_KEY=your_api_here.
- Install: pip install -r requirements.txt
- Launch: python app.py
