from state.job_application_state import JobApplicationState
from typing import List
from utils.text_normalization import normalize_text


# NOTE:
# Simple deterministic chunking used for V1.
# RecursiveCharacterTextSplitter planned for V2
# once embedding quality becomes a bottleneck.

def chunk_text(text: str, chunk_size: int= 500, overlap: int= 30) -> List[str]:
    '''
    Chunk text into smaller pieces with specified chunk size and overlap.
    '''
    if not text:
        return []
    chunks= []
    text_length= len(text)
    start= 0
    while start< text_length:
        end= start+chunk_size
        chunk= text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, 0)

    return chunks

def resume_ingest_node(state: JobApplicationState) -> JobApplicationState: 
    '''
    Resume ingestion node

    Reads:
    resume_raw_text: raw resume text
    Writes:
    normalized_resume_text: Normalized resume text  
    resume_chunks: List of resume text chunks

    '''
    raw_resume_text= state.get("resume_raw_text", "")
    normalized_resume_text= normalize_text(raw_resume_text)
    resume_chunks= chunk_text(normalized_resume_text)   
    state['resume_raw_text']= normalized_resume_text
    state['resume_chunks']= resume_chunks
    return state