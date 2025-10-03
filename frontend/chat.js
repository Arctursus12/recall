// Configuration state
let config = {
    models: [],  // Array of {name, type, model, api_key, url, purpose}
    recallUrl: 'http://localhost:5000'
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
const modelsList = document.getElementById('models-list');
const addModelBtn = document.getElementById('add-model-btn');
const providerTypeSelect = document.getElementById('provider-type');

// Load saved config from localStorage
function loadConfig() {
    const saved = localStorage.getItem('recallConfig');
    if (saved) {
        config = JSON.parse(saved);
        document.getElementById('recall-url').value = config.recallUrl || 'http://localhost:5000';
        renderModelsList();
    }
}

// Save config to localStorage
function saveConfig() {
    config.recallUrl = document.getElementById('recall-url').value;
    localStorage.setItem('recallConfig', JSON.stringify(config));
    showStatus('Configuration saved', 'success');
}

// Render the list of configured models
function renderModelsList() {
    if (config.models.length === 0) {
        modelsList.innerHTML = '<p class="no-models">No models configured. Add a model below.</p>';
        return;
    }

    modelsList.innerHTML = '<h4>Configured Models</h4>';
    config.models.forEach((model, index) => {
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        modelCard.innerHTML = `
            <div class="model-card-header">
                <strong>${model.name}</strong>
                <button class="remove-model-btn" data-index="${index}">Remove</button>
            </div>
            <div class="model-card-details">
                <div><strong>Provider:</strong> ${model.type}</div>
                <div><strong>Model:</strong> ${model.model}</div>
                ${model.url ? `<div><strong>URL:</strong> ${model.url}</div>` : ''}
                <div><strong>Purpose:</strong> ${model.purpose}</div>
            </div>
        `;
        modelsList.appendChild(modelCard);
    });

    // Add event listeners to remove buttons
    document.querySelectorAll('.remove-model-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const index = parseInt(e.target.dataset.index);
            removeModel(index);
        });
    });
}

// Add a new model
function addModel() {
    const name = document.getElementById('model-name').value.trim();
    const type = document.getElementById('provider-type').value;
    const model = document.getElementById('model-id').value.trim();
    const apiKey = document.getElementById('model-api-key').value.trim();
    const url = document.getElementById('model-url').value.trim();
    const purpose = document.getElementById('model-purpose').value.trim();

    // Validation
    if (!name) {
        showStatus('Please enter a model name', 'error');
        return;
    }
    if (!type) {
        showStatus('Please select a provider', 'error');
        return;
    }
    if (!model) {
        showStatus('Please enter a model ID', 'error');
        return;
    }
    if (!purpose) {
        showStatus('Please enter the model\'s primary purpose', 'error');
        return;
    }

    // Add model to config
    const newModel = { name, type, model, purpose };
    if (apiKey) newModel.api_key = apiKey;
    if (url) newModel.url = url;

    config.models.push(newModel);

    // Clear form
    document.getElementById('model-name').value = '';
    document.getElementById('provider-type').value = '';
    document.getElementById('model-id').value = '';
    document.getElementById('model-api-key').value = '';
    document.getElementById('model-url').value = '';
    document.getElementById('model-purpose').value = '';
    document.getElementById('model-url-group').style.display = 'none';

    // Re-render and save
    renderModelsList();
    saveConfig();
    showStatus('Model added successfully', 'success');
}

// Remove a model
function removeModel(index) {
    config.models.splice(index, 1);
    renderModelsList();
    saveConfig();
    showStatus('Model removed', 'success');
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

// Toggle URL field visibility based on provider type
function toggleUrlField() {
    const type = providerTypeSelect.value;
    const urlGroup = document.getElementById('model-url-group');
    const localProviders = ['ollama', 'koboldcpp', 'openai-compatible'];

    if (localProviders.includes(type)) {
        urlGroup.style.display = 'block';
    } else {
        urlGroup.style.display = 'none';
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
        use_memory: true,
        models: config.models  // Send all configured models for routing
    };

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
    if (config.models.length === 0) {
        showStatus('Please configure at least one model', 'error');
        return;
    }

    // Disable input while processing
    userInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message to UI
    addMessage('user', userMessage);
    userInput.value = '';

    try {
        // Send to Recall backend (will route to appropriate model)
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
addModelBtn.addEventListener('click', addModel);
providerTypeSelect.addEventListener('change', toggleUrlField);

// Initialize
loadConfig();
addMessage('system', 'Recall chat initialized. Configure your models and start chatting.');
