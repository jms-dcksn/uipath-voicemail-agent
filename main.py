from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel
from utils.download import download_file
from typing import TypedDict, Optional, Annotated
import whisper


class State(TypedDict):
    file_path: str
    transcription: str
    extracted_data: dict
    summary: str

class GraphInput(BaseModel):
    file_path: str

class GraphOutput(BaseModel):
    extracted_data: dict
    summary: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def transcribe_audio(state: GraphInput) -> State:
    local_file = download_file(state.file_path)
    model = whisper.load_model("base")
    result = model.transcribe(local_file)
    state["transcription"] = result["text"]
    return state

def extract_data(state: State) -> State:
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are an insurance claims assistant. Extract the policy number, incident date, location, and damage description from the following transcript."),
          ("human", f"{state.transcription}")
      ])
      chain = prompt | llm
      response = chain.invoke({"transcription": state["transcription"]})
      state["extracted_data"] = response.content
      return state

def summarize_incident(state: State) -> GraphOutput:
      prompt = ChatPromptTemplate.from_messages([
          ("system", "Summarize the following insurance claim transcript into a concise overview."),
          ("human", "{transcription}")
      ])
      chain = prompt | llm
      response = chain.invoke({"transcription": state["transcription"]})
      return GraphOutput(extracted_data=state.extracted_data, summary=response.content)


builder = StateGraph(input=GraphInput, output=GraphOutput)
builder.add_node("transcribe_audio", transcribe_audio)
builder.add_node("extract_data", extract_data)
builder.add_node("summarize_incident", summarize_incident)

builder.add_edge(START, "transcribe_audio")
builder.add_edge("transcribe_audio", "extract_data")
builder.add_edge("extract_data", "summarize_incident")
builder.add_edge("summarize_incident", END)

graph = builder.compile()