# Recall
<sup>(Name STC)</sup><br>
Memory management system for creating a persistent "AI" with highly personalized functionality, and perpetual continuity across tasks.<br>

## The "What"
Every conversation is automatically saved and semantically indexed, with any identifying factors stored as "reference points." As you speak, Recall will retrieve relevant context from past conversations and inject it into the LLM's prompt, allowing proper continuity and persistance.<br>
The goal long-term is to permit it access to the host system and internet, so that it can attempt to retrieve information it may not even have yet.<br>

## The "How"
Recall uses a **hybrid memory architecture**:<br>

1. **SQLite Database** - Stores conversation metadata, timestamps, and session information for fast chronological queries
2. **ChromaDB Vector Store** - Maintains semantic embeddings of conversations for similarity-based retrieval
3. **Memory Manager** - Retrieves both recent conversations (from your current session) and semantically relevant past conversations (from previous sessions)
4. **Backend Server** - Orchestrates memory operations and routes requests to your chosen LLM (OpenAI or local)
5. **Web Frontend** - Simple chat interface for configuration and interaction

**Memory is always on.** Every message is saved and becomes part of the searchable history. When you send a message, Recall automatically finds the most relevant past conversations and includes them in the LLM's context window. This is vital, as even "useless" or "irrelevant" data shapes how interaction occurs, and ends up contributing to memory in a beneficial way.<br>

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running
```bash
# Start the Recall backend (required for all operations)
python fetcher/main.py

# Backend runs on http://localhost:5000
# Open frontend/chat.html in your browser
```

### Configuration
Choose your LLM in the frontend interface:<br>

**Option 1: OpenAI**
- Enter your API key in the UI (or set `OPENAI_API_KEY` environment variable)
- Select model: GPT-4, GPT-4 Turbo, or GPT-3.5 Turbo
- Messages route through Recall backend → OpenAI API (with memory context)

**Option 2: Local LLM**
- Supports: Ollama, KoboldCpp, LM Studio, or any OpenAI-compatible API
- Configure in UI:
  - **URL**: Your LLM server address (e.g., `http://localhost:11434`)
  - **Type**: Select your server type from dropdown
  - **Model**: Model name (e.g., `llama2`, `mistral`, `codellama`)
- Messages route through Recall backend → Local LLM (with memory context)

**Example Setup with Ollama:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Recall backend
python fetcher/main.py

# Terminal 3: Pull a model if needed
ollama pull llama2

# Open frontend/chat.html, select "Local LLM", configure URL and model
```

## File Structure
```
recall/
├── frontend/           # Web UI (HTML/CSS/JS)
├── database/           # SQLite schema and managers
├── fetcher/            # FastAPI backend + memory logic
├── config/             # (Future) Personality configs
└── requirements.txt    # Python dependencies
```

## API Endpoints
The Recall backend exposes these endpoints:<br>

- `POST /chat` - Main chat endpoint (handles memory retrieval, LLM calls, and storage)
- `POST /memory/search` - Search past conversations by text or semantic similarity
- `GET /memory/recent` - Retrieve recent conversations (optionally filtered by session)
- `GET /sessions` - List all conversation sessions
- `GET /health` - Health check with vector store statistics

## Technical Details

**Memory Retrieval Strategy:**
- Fetches last 10 messages from current session (chronological context)
- Performs semantic search for 3 most relevant past conversations (cross-session context)
- Combines both into system prompt for LLM

**Embedding Model:**
- Uses `all-MiniLM-L6-v2` (80MB, fast, 384 dimensions)
- Runs locally via sentence-transformers
- Alternative: Upgrade to `all-mpnet-base-v2` for higher quality

**Storage:**
- SQLite database: `database/recall.db`
- ChromaDB data: `database/chroma/`
- Both created automatically on first run

**Ports:**
- Recall backend: `5000`
- Ollama default: `11434`
- Other local LLMs: User-configurable<br>

# PR Notes

## Goal
The goal with Recall is to create a framework that will work in perpetuity with newly developing language-oriented "AI" systems.<br>
At the moment, that means LLMs, but with organizations such as [Real Good AI](https://www.realgoodai.org) seeking to innovate in the field, the design intent with Recall is to have it be compatible with future **and** past technologies in perpetuity, and to allow existing configurations of it to be upgraded to keep up with technological advancement.<br>

## Feedback
I am not a skilled programmer, and this project is being developed in parallel to many other obligations and hobbies ongoing in my life.<br>
<sup>(At least well over halfm and likely most, of this code will be written with heavy Claude Code and GPT usage.)</sup><br>
As such, updates and development will likely be slow, features will be broken, and many issues will likely pass by me unnoticed. As such, if you experience any issues or notice any flaws in my work, feel free to open an issue! I'll fix bugs as fast as I physically can, and would very much appreciate advice from more experienced peers in the field. Scrapping together something semi-functional is easy enough, but refining it from a stick and stone into a proper tool requires knowledge I, at least so far, sorely lack.<br>

## Contribution
While this is primarily a personal passion project, contributions are welcome!<br>
Bear in mind there is no guarantee your addition or edit will be implemented – if you want to help, it's best to provide **feedback** instead.<br>

## License
At least for the time being, Recall is being licensed under **Apache 2.0**. This is primarily to future-proof, for the slim circumstance that it picks up steam and potentially encounters any friction.<br>
Recall as a piece of software, at the very least until it is "complete," will be **FOSS**, and will remain so permanently. Whether or not I will at later points allow monetization of forks or derivatives of the project will depend on whether or not it ever reaches that point, when it reaches that point, and in what state it does so.<br>
