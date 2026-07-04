from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the class we created in ai_pipeline.py
from LLM import YouTubeAIPipeline

app = FastAPI()

# Enable CORS for Chrome Extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize a single global instance of your AI helper
ai_helper = YouTubeAIPipeline()

# Pydantic Schemas for data validation
class VideoRequest(BaseModel):
    url: str

class QuestionRequest(BaseModel):
    question: str

@app.post("/process-video")
async def process_video(request: VideoRequest):
    try:
        ai_helper.process_video(request.url)
        return {"status": "success", "message": "Video processed and indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    try:
        answer = ai_helper.ask_question(request.question)
        return {"answer": answer}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))