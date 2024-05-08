from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI # import from chat_models when creating a chatbot app
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="An API Server"
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model = ChatOpenAI()
## llama2
llm = Ollama(model="llama2")

prompt1 = ChatPromptTemplate.from_template("Write an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write an poem about {topic} with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__ =="__main__":
    uvicorn.run(app,host="localhost",port=8000)