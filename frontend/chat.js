// Configuration state
let config = {
    mode: 'openai',
    apiKey: '',
    model: 'gpt-4',
    recallUrl: 'http://localhost:5000',
    llmUrl: 'http://localhost:11434',
    llmType: 'ollama',
    llmModel: ''
};

// Session ID for memory persistence
let sessionId = null;

// DOM elements
const messagesContainer = document.getElementById('messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat');
const saveConfigBtn = document.getElementById('save-config');
const statusDiv = document.getElementById('status');
const modeRadios = document.querySelectorAll('input[name="mode"]');
const openaiConfig = document.getElementById('openai-config');
const localLlmConfig = document.getElementById('local-llm-config');

// Load saved config from localStorage
function loadConfig() {
    const saved = localStorage.getItem('recallConfig');
    if (saved) {
        config = JSON.parse(saved);
        const modeInput = document.querySelector(`input[value="${config.mode}"]`);
        if (modeInput) modeInput.checked = true;

        document.getElementById('api-key').value = config.apiKey || '';
        document.getElementById('model-select').value = config.model || 'gpt-4';
        document.getElementById('recall-url').value = config.recallUrl || 'http://localhost:5000';
        document.getElementById('llm-url').value = config.llmUrl || 'http://localhost:11434';
        document.getElementById('llm-type').value = config.llmType || 'ollama';
        document.getElementById('llm-model').value = config.llmModel || '';

        toggleConfigMode();
    }
}

// Save config to localStorage
function saveConfig() {
    config.mode = document.querySelector('input[name="mode"]:checked').value;
    config.apiKey = document.getElementById('api-key').value;
    config.model = document.getElementById('model-select').value;
    config.recallUrl = document.getElementById('recall-url').value;
    config.llmUrl = document.getElementById('llm-url').value;
    config.llmType = document.getElementById('llm-type').value;
    config.llmModel = document.getElementById('llm-model').value;

    localStorage.setItem('recallConfig', JSON.stringify(config));
    showStatus('Configuration saved', 'success');
}

// Show status message
function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    setTimeout(() => {
        statusDiv.textContent = '';
        statusDiv.className = 'status';
    }, 3000);
}

// Toggle config panel based on mode
function toggleConfigMode() {
    const mode = document.querySelector('input[name="mode"]:checked').value;

    openaiConfig.classList.add('hidden');
    localLlmConfig.classList.add('hidden');

    if (mode === 'openai') {
        openaiConfig.classList.remove('hidden');
    } else if (mode === 'local-llm') {
        localLlmConfig.classList.remove('hidden');
    }
}

// Add message to UI
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = role.charAt(0).toUpperCase() + role.slice(1);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Send message to Recall backend (always with memory)
async function sendToRecall(userMessage) {
    const requestBody = {
        message: userMessage,
        session_id: sessionId,
        use_memory: true
    };

    // Add LLM config if using local LLM
    if (config.mode === 'local-llm') {
        requestBody.llm_config = {
            type: config.llmType,
            url: config.llmUrl,
            model: config.llmModel
        };
    } else if (config.mode === 'openai') {
        requestBody.llm_config = {
            type: 'openai',
            api_key: config.apiKey,
            model: config.model
        };
    }

    const response = await fetch(`${config.recallUrl}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const error = await response.text();
        throw new Error(`Recall backend error: ${error}`);
    }

    const data = await response.json();

    // Store session ID for persistence
    if (data.session_id) {
        sessionId = data.session_id;
    }

    return data.response;
}

// Handle send message
async function handleSend() {
    const userMessage = userInput.value.trim();
    if (!userMessage) return;

    // Validate config
    if (config.mode === 'openai' && !config.apiKey) {
        showStatus('Please enter your OpenAI API key', 'error');
        return;
    }

    if (config.mode === 'local-llm' && !config.llmModel) {
        showStatus('Please enter a model name for local LLM', 'error');
        return;
    }

    // Disable input while processing
    userInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message to UI
    addMessage('user', userMessage);
    userInput.value = '';

    try {
        // All messages go through Recall backend (with memory)
        const assistantMessage = await sendToRecall(userMessage);
        addMessage('assistant', assistantMessage);
    } catch (error) {
        console.error('Error:', error);
        addMessage('system', `Error: ${error.message}`);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

// Clear chat (visual only, memory persists in backend)
function clearChat() {
    messagesContainer.innerHTML = '';
    addMessage('system', 'Chat cleared (memory persists in backend)');
}

// Event listeners
sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});
clearChatBtn.addEventListener('click', clearChat);
saveConfigBtn.addEventListener('click', saveConfig);
modeRadios.forEach(radio => {
    radio.addEventListener('change', toggleConfigMode);
});

// Initialize
loadConfig();
addMessage('system', 'Recall chat initialized. Configure your settings and start chatting.');
