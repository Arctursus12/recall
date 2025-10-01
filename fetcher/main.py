from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import openai
import os
import httpx

from memory_manager import MemoryManager

app = FastAPI(title="Recall API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory manager
memory = MemoryManager()

# Request/Response models
class LLMConfig(BaseModel):
    type: str  # 'openai', 'ollama', 'koboldcpp', 'openai-compatible'
    api_key: Optional[str] = None  # For OpenAI
    model: Optional[str] = None
    url: Optional[str] = None  # For local LLMs

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_memory: bool = True
    llm_config: Optional[LLMConfig] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class MemorySaveRequest(BaseModel):
    session_id: str
    user_message: str
    assistant_response: str
    metadata: Optional[Dict] = None

@app.get("/")
async def root():
    return {
        "name": "Recall API",
        "version": "1.0.0",
        "status": "running"
    }

async def call_llm(messages: List[Dict], llm_config: Optional[LLMConfig]) -> str:
    """
    Call LLM based on config - supports OpenAI, Ollama, KoboldCpp, etc.
    """
    # Use provided config or fall back to environment
    if not llm_config:
        llm_config = LLMConfig(
            type='openai',
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4")
        )

    if llm_config.type == 'openai':
        # OpenAI API
        api_key = llm_config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key required")

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=llm_config.model or "gpt-4",
            messages=messages
        )
        return response.choices[0].message.content

    elif llm_config.type == 'ollama':
        # Ollama API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{llm_config.url}/api/chat",
                json={
                    "model": llm_config.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")
            data = response.json()
            return data["message"]["content"]

    elif llm_config.type == 'koboldcpp':
        # KoboldCpp API (text completion)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{llm_config.url}/api/v1/generate",
                json={"prompt": prompt, "max_length": 500},
                timeout=120.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"KoboldCpp error: {response.text}")
            data = response.json()
            return data["results"][0]["text"].strip()

    elif llm_config.type == 'openai-compatible':
        # OpenAI-compatible API (LM Studio, etc.)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{llm_config.url}/v1/chat/completions",
                json={
                    "model": llm_config.model or "local-model",
                    "messages": messages
                },
                timeout=120.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Local LLM error: {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM type: {llm_config.type}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    - Retrieves relevant memories (always enabled)
    - Sends to configured LLM (OpenAI or local)
    - Saves exchange to memory
    """
    # Create or use existing session
    session_id = request.session_id or memory.create_session()

    # Retrieve context from memory
    context = memory.retrieve_context(
        current_message=request.message,
        session_id=session_id,
        recent_limit=10,
        semantic_limit=3
    )

    # Build messages for LLM
    messages = []

    # Add system prompt with memory context
    system_prompt = "You are a helpful AI assistant with persistent memory."
    if context and (context['recent'] or context['relevant']):
        memory_context = memory.format_context_for_llm(context)
        system_prompt += f"\n\nHere is relevant context from previous conversations:\n{memory_context}"

    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": request.message})

    # Call LLM
    try:
        assistant_message = await call_llm(messages, request.llm_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Save to memory
    memory.save_exchange(
        session_id=session_id,
        user_message=request.message,
        assistant_response=assistant_message
    )

    return ChatResponse(
        response=assistant_message,
        session_id=session_id
    )

@app.post("/memory/save")
async def save_memory(request: MemorySaveRequest):
    """Explicitly save a conversation exchange"""
    conv_id = memory.save_exchange(
        session_id=request.session_id,
        user_message=request.user_message,
        assistant_response=request.assistant_response,
        metadata=request.metadata
    )
    return {"status": "saved", "conversation_id": conv_id}

@app.post("/memory/search")
async def search_memory(request: SearchRequest):
    """Search through memories"""
    results = memory.search_memories(request.query, limit=request.limit)
    return {"results": results}

@app.get("/memory/recent")
async def get_recent(session_id: Optional[str] = None, limit: int = 20):
    """Get recent conversations"""
    conversations = memory.db.get_recent_conversations(
        session_id=session_id,
        limit=limit
    )
    return {"conversations": conversations}

@app.get("/sessions")
async def get_sessions():
    """Get all sessions"""
    sessions = memory.db.get_all_sessions()
    return {"sessions": sessions}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "vector_store_count": memory.vector_store.get_count()
    }

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    memory.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
