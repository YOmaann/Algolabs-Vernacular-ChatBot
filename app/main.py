from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict
from chatbot import MyHelpfulBot
import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    messages: list[str]
    thread_id: str


sessions = defaultdict(lambda: None)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
    
@app.post("/chat")
async def chat(input: ChatInput):
    # config = {"configurable": {"thread_id": input.thread_id}}
    thread_id = input.thread_id

    # Check if ID already initiated
    agent = sessions[thread_id]
    if sessions[thread_id] is None:
        agent = MyHelpfulBot()
        sessions[thread_id] = agent


    responses = []
    for m in input.messages:
        response = await agent.achat(m)
        responses.append(response)
    return responses
