from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel, Field
from utils.download import download_file
from typing import TypedDict, Optional, Union
import openai as whisper
import os
from pathlib import Path


class State(TypedDict):
    file_path: str
    transcription: str
    extracted_data: str
    summary: str

class ExtractedData(BaseModel):
    policy_number: Optional[str] = None
    incident_date: Optional[str] = None
    location: Optional[str] = None
    damage_description: Optional[str] = None

class GraphInput(BaseModel):
    file_path: str
    bucket_name: str
    folder_path: str

class GraphOutput(BaseModel):
    policy_number: Optional[str] = Field(default=None, description="Policy number from the claim")
    incident_date: Optional[str] = Field(default=None, description="Date of the incident")
    location: Optional[str] = Field(default=None, description="Location of the incident")
    damage_description: Optional[str] = Field(default=None, description="Description of the damage")
    summary: str

    @classmethod
    def from_extracted_data(cls, extracted_data: ExtractedData, summary: str) -> "GraphOutput":
        return cls(
            policy_number=extracted_data.policy_number,
            incident_date=extracted_data.incident_date,
            location=extracted_data.location,
            damage_description=extracted_data.damage_description,
            summary=summary
        )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def get_temp_path(filename: str) -> str:
    """Get a temporary path for downloaded files."""
    temp_dir = Path("/tmp")
    return str(temp_dir / filename)

def transcribe_audio(state: GraphInput) -> State:
    try:
        # Use a temporary path for the downloaded file
        temp_path = get_temp_path("voicemail.wav")
        local_file = download_file(
            blob_file_path=state.file_path,
            name=state.bucket_name,
            folder_path=state.folder_path,
            destination_path=temp_path
        )
        print(f'local_file: {local_file}')
        
        # Transcribe the audio
        # Open the audio file
        with open(local_file, "rb") as audio_file:
            # Transcribe the audio file using Whisper
            transcription = whisper.audio.transcriptions.create(
                model="gpt-4o-transcribe", 
                file=audio_file
                )
        
        # Save transcription to a text file
        transcription_path = os.path.splitext(local_file)[0] + "_transcription.txt"
        with open(transcription_path, "w") as f:
            f.write(transcription.text)
        print(f'Transcription saved to: {transcription_path}')
        
        # Clean up the temporary file
        if os.path.exists(local_file):
            os.remove(local_file)
            
        return {
            "file_path": state.file_path,
            "transcription": transcription.text,
            "extracted_data": {},
            "summary": ""
        }
    except Exception as e:
        print(f"Error in transcribe_audio: {str(e)}")
        raise

def extract_data(state: State) -> State:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
                '''You are an insurance claims assistant. Extract the policy number, incident date, location, and damage description from the following transcript. 
                Return the data in JSON format with these exact keys: policy_number, incident_date, location, damage_description.
                If any information is not found, set that field to null.'''),
            ("human", "{transcription}")
        ])
        chain = prompt | llm
        response = chain.invoke({"transcription": state["transcription"]})
        
        # Parse the JSON response into ExtractedData
        extracted_data = ExtractedData.model_validate_json(response.content)
        
        return {
            **state,
            "extracted_data": extracted_data
        }
    except Exception as e:
        print(f"Error in extract_data: {str(e)}")
        raise

def summarize_incident(state: State) -> GraphOutput:
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following insurance claim transcript into a concise overview."),
            ("human", "{transcription}")
        ])
        chain = prompt | llm
        response = chain.invoke({"transcription": state["transcription"]})
        
        extracted = state["extracted_data"]
        return GraphOutput(
            policy_number=extracted.policy_number,
            incident_date=extracted.incident_date,
            location=extracted.location,
            damage_description=extracted.damage_description,
            summary=response.content
        )
    except Exception as e:
        print(f"Error in summarize_incident: {str(e)}")
        raise

# Create the graph
builder = StateGraph(input=GraphInput, output=GraphOutput)
builder.add_node("transcribe_audio", transcribe_audio)
builder.add_node("extract_data", extract_data)
builder.add_node("summarize_incident", summarize_incident)

# Set up the graph flow
builder.add_edge(START, "transcribe_audio")
builder.add_edge("transcribe_audio", "extract_data")
builder.add_edge("extract_data", "summarize_incident")
builder.add_edge("summarize_incident", END)

# Compile the graph
graph = builder.compile()