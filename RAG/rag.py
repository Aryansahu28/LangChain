from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.prompts import PromptTemplate
import time

video_id="FwjaHCVNBWA"
begin_time = int(time.time())
print("Start time",begin_time,"\n\n")
try:
    # If you don’t care which language, this returns the “best” one
    transcript_obj= YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    
    transcript = " ".join(chunk.text for chunk in transcript_obj)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    collection_name="docs_collections",
    embedding=embeddings,
    persist_directory="./RAG/chroma_langchain_db"
)

retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={
        "k": 4
        }
)

model = ChatOllama(
    model="llama3.1:8b"
)

template = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "Name all the topics of PowerBI covered in this"


retrieved_docs =retriever.invoke(question)

context_docs = "\n\n ".join(docs.page_content for docs in retrieved_docs)

final_prompt = template.invoke({"context":context_docs,"question":question})

answer = model.invoke(final_prompt)

print(answer)
end_time = int(time.time())
print("Start time",end_time,"\n\n")

print("difference in time",end_time-begin_time)
