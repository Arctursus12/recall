from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import openai
import os
import httpx

from memory_manager import MemoryManager
from embeddings import get_embedding_generator
import numpy as np

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
    type: str  # 'openai', 'ollama', 'koboldcpp', 'openai-compatible', 'deepseek', 'anthropic', 'google', 'horde'
    api_key: Optional[str] = None  # For OpenAI
    model: Optional[str] = None
    url: Optional[str] = None  # For local LLMs
    purpose: Optional[str] = None  # User-defined purpose for routing

class ModelConfig(BaseModel):
    """Configuration for a single model with purpose"""
    name: str  # User-friendly name
    type: str  # Provider type
    model: str  # Model identifier
    api_key: Optional[str] = None
    url: Optional[str] = None
    purpose: str  # Primary purpose description

class MultiModelConfig(BaseModel):
    """Configuration for multiple models"""
    models: List[ModelConfig]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_memory: bool = True
    llm_config: Optional[LLMConfig] = None  # Legacy single model
    models: Optional[List[ModelConfig]] = None  # New multi-model config

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

def route_to_model(user_message: str, models: List[ModelConfig]) -> ModelConfig:
    """
    Use semantic similarity to route user message to the most appropriate model.

    Uses the existing embedding model (all-MiniLM-L6-v2) to:
    1. Embed the user's message
    2. Embed each model's purpose description
    3. Calculate cosine similarity
    4. Return the model with highest similarity score
    """
    if len(models) == 1:
        return models[0]

    embedder = get_embedding_generator()

    # Generate embedding for user message
    message_embedding = np.array(embedder.generate(user_message))

    # Generate embeddings for all model purposes
    purpose_texts = [model.purpose for model in models]
    purpose_embeddings = embedder.generate_batch(purpose_texts)

    # Calculate cosine similarities
    similarities = []
    for purpose_emb in purpose_embeddings:
        purpose_emb = np.array(purpose_emb)
        similarity = np.dot(message_embedding, purpose_emb) / (
            np.linalg.norm(message_embedding) * np.linalg.norm(purpose_emb)
        )
        similarities.append(float(similarity))

    # Return model with highest similarity
    best_idx = int(np.argmax(similarities))
    return models[best_idx]

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

    elif llm_config.type == 'deepseek':
        # Deepseek API (OpenAI-compatible)
        api_key = llm_config.api_key
        if not api_key:
            raise HTTPException(status_code=500, detail="Deepseek API key required")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": llm_config.model or "deepseek-chat",
                    "messages": messages
                },
                timeout=120.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Deepseek error: {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]

    elif llm_config.type == 'anthropic':
        # Anthropic Claude API
        api_key = llm_config.api_key
        if not api_key:
            raise HTTPException(status_code=500, detail="Anthropic API key required")

        # Convert OpenAI format to Anthropic format
        system_content = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                anthropic_messages.append(msg)

        async with httpx.AsyncClient() as client:
            request_body = {
                "model": llm_config.model or "claude-3-5-sonnet-20241022",
                "messages": anthropic_messages,
                "max_tokens": 4096
            }
            if system_content:
                request_body["system"] = system_content

            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=request_body,
                timeout=120.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Anthropic error: {response.text}")
            data = response.json()
            return data["content"][0]["text"]

    elif llm_config.type == 'google':
        # Google Gemini API
        api_key = llm_config.api_key
        if not api_key:
            raise HTTPException(status_code=500, detail="Google API key required")

        model = llm_config.model or "gemini-pro"

        # Convert OpenAI format to Gemini format
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": gemini_contents},
                timeout=120.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Google error: {response.text}")
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    elif llm_config.type == 'horde':
        # AI Horde (KoboldAI Horde) - Free distributed inference
        async with httpx.AsyncClient() as client:
            # Convert messages to prompt
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

            # Submit generation request
            submit_response = await client.post(
                "https://stablehorde.net/api/v2/generate/text/async",
                headers={"apikey": llm_config.api_key or "0000000000"},  # Anonymous key
                json={
                    "prompt": prompt,
                    "params": {
                        "max_length": 500,
                        "max_context_length": 2048
                    },
                    "models": [llm_config.model] if llm_config.model else []
                },
                timeout=30.0
            )
            if submit_response.status_code != 202:
                raise HTTPException(status_code=500, detail=f"Horde submit error: {submit_response.text}")

            submit_data = submit_response.json()
            request_id = submit_data["id"]

            # Poll for completion (max 2 minutes)
            import asyncio
            for _ in range(24):  # 24 * 5s = 2 minutes
                await asyncio.sleep(5)
                status_response = await client.get(
                    f"https://stablehorde.net/api/v2/generate/text/status/{request_id}",
                    timeout=10.0
                )
                if status_response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"Horde status error: {status_response.text}")

                status_data = status_response.json()
                if status_data["done"]:
                    return status_data["generations"][0]["text"].strip()

            raise HTTPException(status_code=504, detail="Horde request timed out")

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM type: {llm_config.type}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    - Routes to appropriate model based on semantic similarity (if multi-model config provided)
    - Retrieves relevant memories (always enabled)
    - Sends to configured LLM
    - Saves exchange to memory
    """
    # Create or use existing session
    session_id = request.session_id or memory.create_session()

    # Determine which LLM config to use
    llm_config = request.llm_config

    # If multi-model config provided, route to best model
    if request.models and len(request.models) > 0:
        selected_model = route_to_model(request.message, request.models)

        # Convert ModelConfig to LLMConfig
        llm_config = LLMConfig(
            type=selected_model.type,
            model=selected_model.model,
            api_key=selected_model.api_key,
            url=selected_model.url,
            purpose=selected_model.purpose
        )

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
        assistant_message = await call_llm(messages, llm_config)
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
