// PCM Audio Worklet for VERA Cloud
// Converts Float32 audio to Int16 PCM and sends 20-30ms frames
// Optimized for Azure cloud deployment

class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // Audio processing parameters
        this.sampleRate = 16000;
        this.frameSize = Math.floor(this.sampleRate * 0.03); // 30ms frames
        this.inputBuffer = new Float32Array(this.frameSize);
        this.bufferIndex = 0;
        this.isActive = true;
        
        // Level meter for UI feedback
        this.levelSmoothing = 0.95;
        this.smoothedLevel = 0;
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastLevelUpdate = 0;
        this.levelUpdateInterval = 100; // Update level every 100ms
        
        console.log(`PCMProcessor initialized: ${this.frameSize} samples per frame at ${this.sampleRate}Hz`);
    }
    
    process(inputs, outputs, parameters) {
        if (!this.isActive) {
            return false;
        }
        
        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }
        
        // Get the first channel (mono)
        const inputChannel = input[0];
        if (!inputChannel) {
            return true;
        }
        
        // Compute level meter (RMS) for UI feedback
        const currentTime = currentFrame / this.sampleRate * 1000; // Convert to milliseconds
        if (currentTime - this.lastLevelUpdate >= this.levelUpdateInterval) {
            let sumSquares = 0;
            for (let i = 0; i < inputChannel.length; i++) {
                const s = inputChannel[i];
                sumSquares += s * s;
            }
            const rms = Math.sqrt(sumSquares / inputChannel.length);
            
            // Smooth the level for better UI experience
            this.smoothedLevel = this.smoothedLevel * this.levelSmoothing + rms * (1 - this.levelSmoothing);
            const level = Math.min(1, this.smoothedLevel * 3); // Scale for visibility
            
            this.port.postMessage({ 
                type: 'level', 
                value: level,
                timestamp: currentTime
            });
            
            this.lastLevelUpdate = currentTime;
        }

        // Process each sample in the input
        for (let i = 0; i < inputChannel.length; i++) {
            // Add sample to buffer
            this.inputBuffer[this.bufferIndex] = inputChannel[i];
            this.bufferIndex++;
            
            // When buffer is full, send frame
            if (this.bufferIndex >= this.frameSize) {
                this.sendFrame();
                this.bufferIndex = 0;
            }
        }
        
        return true;
    }
    
    sendFrame() {
        // Convert Float32 to Int16 PCM
        const pcmFrame = new Int16Array(this.frameSize);
        
        for (let i = 0; i < this.frameSize; i++) {
            // Clamp and convert to 16-bit PCM
            const sample = Math.max(-1, Math.min(1, this.inputBuffer[i]));
            pcmFrame[i] = Math.round(sample * 32767);
        }
        
        // Send the PCM frame
        this.port.postMessage({
            type: 'audio',
            data: pcmFrame.buffer,
            timestamp: currentFrame / this.sampleRate * 1000,
            frameSize: this.frameSize,
            sampleRate: this.sampleRate
        });
        
        this.frameCount++;
        
        // Log performance every 1000 frames (about 30 seconds)
        if (this.frameCount % 1000 === 0) {
            console.log(`PCMProcessor: Processed ${this.frameCount} frames`);
        }
    }
    
    // Handle messages from main thread
    static get parameterDescriptors() {
        return [];
    }
}

// Register the processor
registerProcessor('pcm-processor', PCMProcessor);
