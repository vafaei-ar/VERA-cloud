// VERA Cloud - Frontend JavaScript Application
// Optimized for Azure cloud deployment

class VERACloudApp {
    constructor() {
        this.sessionId = null;
        this.websocket = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.isRecording = false;
        this.isConnected = false;
        this.currentScenario = null;
        this.ragEnabled = false;
        
        // Audio recording
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isAudioRecording = false;
        
        // UI Elements
        this.elements = {
            setup: document.getElementById('setup'),
            conversation: document.getElementById('conversation'),
            honorific: document.getElementById('honorific'),
            patientName: document.getElementById('patientName'),
            scenario: document.getElementById('scenario'),
            voice: document.getElementById('voice'),
            rate: document.getElementById('rate'),
            rateValue: document.querySelector('.rate-value'),
            startButton: document.getElementById('startButton'),
            endButton: document.getElementById('endButton'),
            downloadButton: document.getElementById('downloadButton'),
            micButton: document.getElementById('micButton'),
            micLevel: document.getElementById('micLevel'),
            micStatus: document.getElementById('micStatus'),
            transcript: document.getElementById('transcript'),
            textInput: document.getElementById('textInput'),
            sendTextButton: document.getElementById('sendTextButton'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            connectionStatus: document.getElementById('connectionStatus'),
            ragContext: document.getElementById('ragContext'),
            ragContent: document.getElementById('ragContent'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            errorModal: document.getElementById('errorModal'),
            errorMessage: document.getElementById('errorMessage')
        };
        
        this.init();
    }
    
    async init() {
        try {
            this.setupEventListeners();
            await this.loadScenarios();
            await this.loadVoices();
            this.updateConnectionStatus(false);
            this.showSection('setup');
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showError('Failed to initialize application');
        }
    }
    
    setupEventListeners() {
        // Form controls
        this.elements.startButton.addEventListener('click', () => this.startCall());
        this.elements.endButton.addEventListener('click', () => this.endCall());
        this.elements.downloadButton.addEventListener('click', () => this.downloadTranscript());
        
        // Rate slider
        this.elements.rate.addEventListener('input', (e) => {
            this.elements.rateValue.textContent = `${parseFloat(e.target.value).toFixed(1)}x`;
        });
        
        // Microphone button
        this.elements.micButton.addEventListener('click', () => this.toggleRecording());
        
        // Text input fallback
        this.elements.sendTextButton.addEventListener('click', () => this.sendTextInput());
        this.elements.textInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendTextInput();
            }
        });
        
        // Modal controls
        document.getElementById('errorModalClose').addEventListener('click', () => this.hideError());
        document.getElementById('errorModalOk').addEventListener('click', () => this.hideError());
        
        // Scenario change
        this.elements.scenario.addEventListener('change', (e) => {
            this.currentScenario = e.target.value;
            this.updateStartButton();
        });
        
        // Form validation
        [this.elements.honorific, this.elements.scenario].forEach(element => {
            element.addEventListener('change', () => this.updateStartButton());
        });
    }
    
    async loadScenarios() {
        try {
            const response = await fetch('/api/scenarios');
            const data = await response.json();
            
            this.elements.scenario.innerHTML = '<option value="">Select conversation mode...</option>';
            
            data.scenarios.forEach(scenario => {
                const option = document.createElement('option');
                option.value = scenario.filename;
                option.textContent = `${scenario.name} (${scenario.mode})`;
                this.elements.scenario.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load scenarios:', error);
            this.elements.scenario.innerHTML = '<option value="">Error loading scenarios</option>';
        }
    }
    
    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            const data = await response.json();
            
            this.elements.voice.innerHTML = '<option value="">Default voice</option>';
            
            Object.entries(data.voices).forEach(([voiceId, voiceInfo]) => {
                const option = document.createElement('option');
                option.value = voiceId;
                option.textContent = `${voiceId} (${voiceInfo.gender}, ${voiceInfo.style})`;
                this.elements.voice.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load voices:', error);
        }
    }
    
    updateStartButton() {
        const canStart = this.elements.honorific.value && this.elements.scenario.value;
        this.elements.startButton.disabled = !canStart;
    }
    
    async startCall() {
        try {
            this.showLoading('Starting conversation...');
            
            // Initialize audio
            await this.initializeAudio();
            
            // Start session
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    honorific: this.elements.honorific.value,
                    patient_name: this.elements.patientName.value || 'Patient',
                    scenario: this.elements.scenario.value,
                    voice: this.elements.voice.value || null,
                    rate: parseFloat(this.elements.rate.value)
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const sessionData = await response.json();
            this.sessionId = sessionData.session_id;
            this.ragEnabled = sessionData.mode === 'rag_enhanced';
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Show conversation UI
            this.showSection('conversation');
            this.hideLoading();
            
        } catch (error) {
            console.error('Failed to start call:', error);
            this.hideLoading();
            this.showError(`Failed to start conversation: ${error.message}`);
        }
    }
    
    async initializeAudio() {
        try {
            this.setStatus('Requesting microphone access...');
            
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio context
            this.audioContext = new AudioContext({ sampleRate: 16000 });
            
            // Initialize Speech Recognition
            this.initializeSpeechRecognition();
            
            this.setStatus('Audio initialized');
            
        } catch (error) {
            throw new Error('Microphone access denied or unavailable');
        }
    }
    
    initializeSpeechRecognition() {
        // Check if Speech Recognition is supported
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech Recognition not supported in this browser');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
        
        this.recognition.onstart = () => {
            console.log('Speech recognition started');
            this.setStatus('Listening...');
        };
        
        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            // Show interim results
            if (interimTranscript) {
                this.addTranscriptMessage(interimTranscript, 0.5, false, 'user');
            }
            
            // Send final results to server
            if (finalTranscript) {
                console.log('Final transcript:', finalTranscript);
                this.addTranscriptMessage(finalTranscript, 1.0, true, 'user');
                this.sendTextToServer(finalTranscript);
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.setStatus('Speech recognition error');
        };
        
        this.recognition.onend = () => {
            console.log('Speech recognition ended');
            this.setStatus('Click to speak');
        };
    }
    
    sendTextToServer(text) {
        console.log('Sending text to server:', text);
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'text_input',
                text: text
            };
            console.log('WebSocket message:', message);
            this.websocket.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected:', this.websocket?.readyState);
        }
    }
    
    sendTextInput() {
        const text = this.elements.textInput.value.trim();
        if (text) {
            console.log('Sending text input:', text);
            this.addTranscriptMessage(text, 1.0, true, 'user');
            this.sendTextToServer(text);
            this.elements.textInput.value = ''; // Clear input
        }
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.setStatus('Connecting to server...');
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/audio/${this.sessionId}`;
            
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected successfully');
                this.setStatus('Connected');
                this.updateConnectionStatus(true);
                this.isConnected = true;
                resolve();
            };
            
            this.websocket.onmessage = (event) => {
                console.log('WebSocket message received:', event.data.substring(0, 100) + '...');
                if (event.data instanceof ArrayBuffer) {
                    // Received TTS audio
                    this.playTTSAudio(event.data);
                } else {
                    // Received text message
                    try {
                        const message = JSON.parse(event.data);
                        console.log('Parsed message:', message);
                        this.handleWebSocketMessage(message);
                    } catch (e) {
                        console.log('Server message:', event.data);
                    }
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showError('Connection error. Please try again.');
                reject(error);
            };
            
            this.websocket.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.setStatus('Disconnected');
                this.updateConnectionStatus(false);
                this.isConnected = false;
                if (event.code !== 1000) {
                    this.showError('Connection lost. Please refresh the page.');
                }
            };
            
            // Timeout after 10 seconds
            setTimeout(() => {
                if (this.websocket.readyState !== WebSocket.OPEN) {
                    this.websocket.close();
                    reject(new Error('Connection timeout'));
                }
            }, 10000);
        });
    }
    
    handleWebSocketMessage(message) {
        if (message && message.bot_sequence !== undefined) {
            this.lastBotSequence = message.bot_sequence;
        }
        switch (message.type) {
            case 'transcript_update':
                this.addTranscriptMessage(message.text, message.confidence, message.is_final);
                break;
            case 'audio':
                // Handle TTS audio data
                this.addTranscriptMessage(message.text, 1.0, true, 'bot');
                this.playTTSAudioFromBase64(message.audio_data, message.text);
                break;
            case 'greeting':
                // Handle initial greeting
                this.addTranscriptMessage(message.text, 1.0, true, 'bot');
                if (message.audio_data) {
                    this.playTTSAudioFromBase64(message.audio_data, message.text);
                }
                break;
            case 'response':
                // Handle bot response
                this.addTranscriptMessage(message.text, 1.0, true, 'bot');
                if (message.audio_data) {
                    this.playTTSAudioFromBase64(message.audio_data, message.text);
                }
                break;
            case 'question':
                // Handle bot question
                this.addTranscriptMessage(message.text, 1.0, true, 'bot');
                if (message.audio_data) {
                    this.playTTSAudioFromBase64(message.audio_data, message.text);
                }
                break;
            case 'rag_context':
                this.showRAGContext(message);
                break;
            case 'emergency_alert':
                this.showEmergencyAlert(message);
                break;
            case 'session_summary':
                this.showSessionSummary(message);
                break;
            case 'completion':
                // Handle conversation completion
                this.addTranscriptMessage(message.text, 1.0, true, 'bot');
                if (message.audio_data) {
                    this.playTTSAudioFromBase64(message.audio_data, message.text);
                }
                this.handleConversationComplete();
                break;
            case 'error':
                this.showError(message.message);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
        
        // Update progress bar if available
        if (message.progress !== undefined) {
            console.log('Received progress update:', message.progress);
            this.updateProgress(message.progress);
        } else {
            console.log('No progress in message:', message);
        }
    }
    
    async playTTSAudio(audioData) {
        try {
            if (!this.audioContext) return;
            
            const audioBuffer = await this.audioContext.decodeAudioData(audioData);
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
        } catch (error) {
            console.error('Failed to play TTS audio:', error);
        }
    }
    
    async playTTSAudioFromBase64(base64Data, text) {
        try {
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            // Convert base64 to ArrayBuffer
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
            console.log('Playing TTS audio for:', text);
        } catch (error) {
            console.error('Error playing TTS audio from base64:', error);
        }
    }
    
    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }
    
    startRecording() {
        if (!this.isConnected) {
            this.showError('Not connected to server');
            return;
        }
        
        if (!this.recognition) {
            this.showError('Speech recognition not supported in this browser');
            return;
        }
        
        this.isRecording = true;
        this.elements.micButton.classList.add('recording');
        this.elements.micStatus.textContent = 'Listening...';
        this.setStatus('Listening...');
        
        // Start speech recognition
        this.recognition.start();
        
        // Also start audio recording for storage
        this.startAudioRecording();
    }
    
    stopRecording() {
        this.isRecording = false;
        this.elements.micButton.classList.remove('recording');
        this.elements.micStatus.textContent = 'Click to speak';
        this.setStatus('Click to speak');
        
        // Stop speech recognition
        if (this.recognition) {
            this.recognition.stop();
        }
        
        // Also stop audio recording
        this.stopAudioRecording();
    }
    
    addTranscriptMessage(text, confidence, isFinal, sender = 'user') {
        if (!text.trim()) return;
        
        const messageDiv = document.createElement('div');
        const isBot = sender === 'bot';
        messageDiv.className = `transcript-message ${isBot ? 'bot-message' : 'user-message'} ${isFinal ? 'final' : 'partial'}`;
        
        const confidenceClass = confidence > 0.8 ? 'high' : confidence > 0.5 ? 'medium' : 'low';
        const avatar = isBot ? '🤖' : '👤';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${text}</div>
                <div class="message-meta">
                    <span class="message-time">${new Date().toLocaleTimeString()}</span>
                    ${isFinal ? `<span class="confidence ${confidenceClass}">${Math.round(confidence * 100)}%</span>` : '<span class="partial-indicator">...</span>'}
                </div>
            </div>
        `;
        
        this.elements.transcript.appendChild(messageDiv);
        this.elements.transcript.scrollTop = this.elements.transcript.scrollHeight;
        
        if (isFinal && !isBot) {
            this.stopRecording();
        }
    }
    
    addAIQuestion(text, questionKey) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'transcript-message ai-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="message-text">${text}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        this.elements.transcript.appendChild(messageDiv);
        this.elements.transcript.scrollTop = this.elements.transcript.scrollHeight;
    }
    
    showRAGContext(message) {
        if (!this.ragEnabled) return;
        
        this.elements.ragContext.style.display = 'block';
        this.elements.ragContent.innerHTML = `
            <div class="rag-info">
                <span class="rag-enhanced">✓ RAG Enhanced</span>
                <span class="context-used">Context: ${message.context_used} sources</span>
                <span class="confidence">Confidence: ${Math.round(message.confidence * 100)}%</span>
            </div>
        `;
    }
    
    showEmergencyAlert(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'emergency-alert';
        alertDiv.innerHTML = `
            <div class="alert-icon">🚨</div>
            <div class="alert-content">
                <h3>Emergency Detected</h3>
                <p>${message.message}</p>
                <p><strong>If this is a medical emergency, call 911 immediately.</strong></p>
            </div>
        `;
        
        this.elements.transcript.appendChild(alertDiv);
        this.elements.transcript.scrollTop = this.elements.transcript.scrollHeight;
    }
    
    showSessionSummary(message) {
        this.elements.downloadButton.style.display = 'inline-block';
        this.setStatus('Session completed');
        this.stopRecording();
    }
    
    updateMicLevel(level) {
        const percentage = Math.min(100, level * 100);
        this.elements.micLevel.style.width = `${percentage}%`;
    }
    
    updateConnectionStatus(connected) {
        const indicator = this.elements.connectionStatus.querySelector('.status-indicator');
        const text = this.elements.connectionStatus.querySelector('.status-text');
        
        if (connected) {
            indicator.className = 'status-indicator online';
            text.textContent = 'Connected';
        } else {
            indicator.className = 'status-indicator offline';
            text.textContent = 'Disconnected';
        }
    }
    
    updateProgress(progress) {
        this.elements.progressFill.style.width = `${progress}%`;
    }
    
    setStatus(message) {
        this.elements.progressText.textContent = message;
    }
    
    showSection(sectionName) {
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(sectionName).classList.add('active');
    }
    
    showLoading(message) {
        this.elements.loadingOverlay.querySelector('.loading-text').textContent = message;
        this.elements.loadingOverlay.style.display = 'flex';
    }
    
    hideLoading() {
        this.elements.loadingOverlay.style.display = 'none';
    }
    
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorModal.style.display = 'block';
    }
    
    updateProgress(percentage) {
        console.log('Updating progress to:', percentage);
        
        // Update progress bar if it exists
        const progressBar = document.getElementById('progressBar');
        console.log('Progress bar element:', progressBar);
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
            console.log('Updated progress bar width to:', `${percentage}%`);
        } else {
            console.warn('Progress bar element not found!');
        }
        
        // Update progress text if it exists
        const progressText = document.getElementById('progressText');
        console.log('Progress text element:', progressText);
        if (progressText) {
            progressText.textContent = `Progress: ${Math.round(percentage)}%`;
            console.log('Updated progress text to:', `Progress: ${Math.round(percentage)}%`);
        } else {
            console.warn('Progress text element not found!');
        }
    }
    
    async handleConversationComplete() {
        console.log('Conversation completed!');
        this.setStatus('Conversation completed');
        
        // Fetch conversation data
        try {
            console.log(`Fetching conversation data for session: ${this.sessionId}`);
            const response = await fetch(`/api/conversation-data/${this.sessionId}`);
            const data = await response.json();
            
            console.log('Conversation data received:', data);
            
            if (data.error) {
                console.error('Failed to fetch conversation data:', data.error);
                alert(`Failed to fetch conversation data: ${data.error}`);
                return;
            }
            
            // Check if data is empty
            if (!data.transcript || data.transcript.length === 0) {
                console.warn('No transcript data available');
                alert('No conversation data available for download. This might be because the conversation was not recorded properly.');
            }
            
            if (!data.audio_segments || data.audio_segments.length === 0) {
                console.warn('No audio segments available');
            }
            
            // Show conversation summary
            this.showConversationSummary(data);
            
        } catch (error) {
            console.error('Error fetching conversation data:', error);
            alert(`Error fetching conversation data: ${error.message}`);
        }
    }
    
    showConversationSummary(data) {
        console.log('Showing conversation summary with data:', data);
        
        // Create a modal or section to show the conversation summary
        const summaryHtml = `
            <div class="conversation-summary">
                <h3>Conversation Summary</h3>
                <div class="summary-stats">
                    <p><strong>Duration:</strong> ${Math.round(data.duration_seconds)} seconds</p>
                    <p><strong>Total Messages:</strong> ${data.summary.total_messages}</p>
                    <p><strong>Bot Messages:</strong> ${data.summary.bot_messages}</p>
                    <p><strong>User Messages:</strong> ${data.summary.user_messages}</p>
                    <p><strong>Audio Segments:</strong> ${data.summary.audio_segments}</p>
                </div>
                    <div class="conversation-actions">
                        <button id="downloadTranscriptBtn" class="btn btn-primary">Download Transcript</button>
                        <button id="downloadAudioBtn" class="btn btn-secondary">Download Complete Audio (MP3)</button>
                        <button id="downloadAllAudioBtn" class="btn btn-secondary">Download All Audio Segments (ZIP)</button>
                    </div>
                <div class="debug-info" style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 12px;">
                    <p><strong>Debug Info:</strong></p>
                    <p>Transcript entries: ${data.transcript ? data.transcript.length : 0}</p>
                    <p>Audio segments: ${data.audio_segments ? data.audio_segments.length : 0}</p>
                    <p>Bot audio segments: ${data.audio_segments ? data.audio_segments.filter(s => s.type === 'bot').length : 0}</p>
                    <p>User audio segments: ${data.audio_segments ? data.audio_segments.filter(s => s.type === 'user').length : 0}</p>
                    <p>Session ID: ${data.session_id}</p>
                </div>
            </div>
        `;
        
        // Add to the conversation container
        const conversationContainer = document.querySelector('.conversation-container');
        if (conversationContainer) {
            conversationContainer.insertAdjacentHTML('beforeend', summaryHtml);
        }
        
        // Store conversation data for download
        this.conversationData = data;
        console.log('Stored conversation data for download:', this.conversationData);
        
            // Add event listeners for download buttons
            setTimeout(() => {
                const transcriptBtn = document.getElementById('downloadTranscriptBtn');
                const audioBtn = document.getElementById('downloadAudioBtn');
                const allAudioBtn = document.getElementById('downloadAllAudioBtn');
                
                if (transcriptBtn) {
                    console.log('Adding event listener for transcript button');
                    transcriptBtn.addEventListener('click', () => {
                        console.log('Transcript button clicked via event listener');
                        this.downloadTranscript();
                    });
                } else {
                    console.error('Transcript button not found!');
                }
                
                if (audioBtn) {
                    console.log('Adding event listener for audio button');
                    audioBtn.addEventListener('click', () => {
                        console.log('Audio button clicked via event listener');
                        this.downloadAudio();
                    });
                } else {
                    console.error('Audio button not found!');
                }
                
                if (allAudioBtn) {
                    console.log('Adding event listener for all audio button');
                    allAudioBtn.addEventListener('click', () => {
                        console.log('All audio button clicked via event listener');
                        this.downloadAllAudio();
                    });
                } else {
                    console.error('All audio button not found!');
                }
            }, 100); // Small delay to ensure DOM is updated
    }
    
    downloadTranscript() {
        console.log('Download transcript clicked');
        console.log('Conversation data:', this.conversationData);
        
        if (!this.conversationData) {
            console.error('No conversation data available for download');
            alert('No conversation data available for download');
            return;
        }
        
        const transcript = this.conversationData.transcript.map(entry => 
            `[${entry.timestamp}] ${entry.speaker}: ${entry.text}`
        ).join('\n');
        
        console.log('Generated transcript:', transcript);
        
        const blob = new Blob([transcript], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `conversation-transcript-${this.sessionId}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Transcript download initiated');
    }
    
        async downloadAudio() {
            console.log('Download audio clicked');
            console.log('Conversation data:', this.conversationData);
            
            if (!this.conversationData) {
                console.error('No conversation data available for download');
                alert('No conversation data available for download');
                return;
            }
            
            if (!this.conversationData.audio_segments || !this.conversationData.audio_segments.length) {
                console.error('No audio segments available for download');
                alert('No audio segments available for download');
                return;
            }
            
            console.log(`Found ${this.conversationData.audio_segments.length} audio segments`);
            
            try {
                // Use the server-side endpoint to get concatenated audio (bot only)
                const response = await fetch(`/api/conversation-audio/${this.sessionId}`);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to fetch audio');
                }
                
                // Get the audio blob
                const audioBlob = await response.blob();
                
                // Create download link
                const url = URL.createObjectURL(audioBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `conversation-audio-complete-${this.sessionId}.mp3`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                console.log('Complete conversation audio download initiated');
                
            } catch (error) {
                console.error('Error downloading audio:', error);
                alert(`Error downloading audio: ${error.message}`);
            }
        }
        
        async downloadAllAudio() {
            console.log('Download all audio clicked');
            console.log('Conversation data:', this.conversationData);
            
            if (!this.conversationData) {
                console.error('No conversation data available for download');
                alert('No conversation data available for download');
                return;
            }
            
            if (!this.conversationData.audio_segments || !this.conversationData.audio_segments.length) {
                console.error('No audio segments available for download');
                alert('No audio segments available for download');
                return;
            }
            
            console.log(`Found ${this.conversationData.audio_segments.length} audio segments`);
            
            try {
                // Use the server-side endpoint to get all audio segments as ZIP
                const response = await fetch(`/api/conversation-audio-with-user/${this.sessionId}`);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Failed to fetch audio segments');
                }
                
                // Get the ZIP blob
                const zipBlob = await response.blob();
                
                // Create download link
                const url = URL.createObjectURL(zipBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `conversation-audio-segments-${this.sessionId}.zip`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
                console.log('All audio segments download initiated');
                
            } catch (error) {
                console.error('Error downloading audio segments:', error);
                alert(`Error downloading audio segments: ${error.message}`);
            }
        }
    
    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }
    
    async startAudioRecording() {
        try {
            console.log('Starting audio recording...');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Try to use MP4 format if supported, fallback to WebM
            let mimeType = 'audio/mp4';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/webm';
                console.log('MP4 not supported, using WebM');
            } else {
                console.log('Using MP4 format for recording');
            }
            
            this.mediaRecorder = new MediaRecorder(stream, { mimeType });
            this.audioChunks = [];
            // Capture timing + bot turn when recording starts
            this.recordingStartedAt = Date.now();
            this.recordingBotSequence = this.lastBotSequence ?? null;
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    console.log(`Audio chunk received: ${event.data.size} bytes`);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('Audio recording stopped, processing...');
                this.processAudioRecording();
            };
            
            this.mediaRecorder.start(1000); // Record in 1-second chunks
            this.isAudioRecording = true;
            console.log(`Audio recording started with format: ${mimeType}`);
            // Send a small metadata envelope ahead of the audio so the server can align
            try {
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    const meta = {
                        type: 'audio_meta',
                        bot_sequence: this.recordingBotSequence,
                        started_at_ms: this.recordingStartedAt
                    };
                    this.websocket.send(JSON.stringify(meta));
                    console.log('Sent audio_meta:', meta);
                }
            } catch (e) {
                console.warn('Failed to send audio_meta:', e);
            }
            
        } catch (error) {
            console.error('Error starting audio recording:', error);
            alert('Error starting audio recording: ' + error.message);
        }
    }
    
    stopAudioRecording() {
        if (this.mediaRecorder && this.isAudioRecording) {
            console.log('Stopping audio recording...');
            this.mediaRecorder.stop();
            this.isAudioRecording = false;
            
            // Stop all tracks
            if (this.mediaRecorder.stream) {
                this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
    }
    
        processAudioRecording() {
            if (this.audioChunks.length === 0) {
                console.log('No audio chunks to process');
                return;
            }
            
            // Determine the MIME type from the media recorder
            const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
            const audioBlob = new Blob(this.audioChunks, { type: mimeType });
            
            console.log(`Processed audio recording: ${audioBlob.size} bytes, type: ${mimeType}`);
            console.log(`Audio chunks: ${this.audioChunks.length}`);
            console.log(`WebSocket state: ${this.websocket?.readyState} (1=OPEN, 0=CONNECTING, 2=CLOSING, 3=CLOSED)`);
            
            // Send audio to server via WebSocket
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                console.log('Sending user audio to server via WebSocket...');
                this.websocket.send(audioBlob);
                console.log('Sent user audio to server');
            } else {
                console.error('WebSocket not available for sending audio. State:', this.websocket?.readyState);
                console.error('WebSocket connection status:', this.isConnected);
            }
            
            // Clear chunks for next recording
            this.audioChunks = [];
            this.recordingEndedAt = Date.now();
        }
    
    hideError() {
        this.elements.errorModal.style.display = 'none';
    }
    
    async endCall() {
        try {
            if (this.websocket) {
                this.websocket.close();
            }
            
            if (this.mediaStream) {
                this.mediaStream.getTracks().forEach(track => track.stop());
            }
            
            if (this.audioContext) {
                await this.audioContext.close();
            }
            
            this.showSection('setup');
            this.sessionId = null;
            this.isConnected = false;
            this.isRecording = false;
            
        } catch (error) {
            console.error('Error ending call:', error);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.veraApp = new VERACloudApp();
    window.app = window.veraApp; // Also make it available as 'app' for backward compatibility
});
