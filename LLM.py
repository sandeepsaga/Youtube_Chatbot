from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
class YouTubeAIPipeline:
    def __init__(self):
        self.vector_store = None
        self.final_chain = None

    def _get_video_id(self, url: str) -> str:
        """Extracts the 11-character video ID from a YouTube URL."""
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        return url

    def _format_docs(self, retrieved_docs):
        """Helper to stitch matching chunks into a single string."""
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    def process_video(self, url: str):
        """Fetches transcript, chunks it, embeds it, and builds the chain."""
        video_id = self._get_video_id(url)
        try:
            # If you don’t care which language, this returns the “best” one
            transcript_li = YouTubeTranscriptApi()
            transcript_list=transcript_li.fetch(video_id,languages=['en'],cookies="cookies.txt")

            # Flatten it to plain text
            transcript = " ".join(chunk['text'] for chunk in transcript_list)
            # print(transcript)

        except TranscriptsDisabled:
            print("No captions available for this video.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        self.vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

        prompt = PromptTemplate(
            template="""
              You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.

              {context}
              Question: {question}
            """,
            input_variables = ['context', 'question']
        )

        def format_docs(retrieved_docs):
          context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
          return context_text

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser=StrOutputParser()
        self.final_chain=parallel_chain|prompt|llm|parser

    def ask_question(self, question: str) -> str:
      if not self.final_chain:
          raise ValueError("No video pipeline has been initialized yet.")
      return self.final_chain.invoke(question)