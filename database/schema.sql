-- Conversations table: stores individual message pairs
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    metadata TEXT, -- JSON: {keywords: [...], tags: [...], importance: 0-10}
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Sessions table: groups conversations by session
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
    summary TEXT,
    metadata TEXT -- JSON: {topic: "...", mood: "...", etc}
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active);
