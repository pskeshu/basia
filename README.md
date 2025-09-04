# Basia - MLOps for Local Llama VLM Microscope Control

Basia is an MLOps platform for hosting local Llama Vision Language Models (VLMs) as hybrid orchestrators, specifically designed for microscope control applications. It enables multiple clients to connect to different sessions of the Llama model simultaneously for real-time image analysis and equipment control.

## Features

- **Microscope Integration**: Real-time image analysis from microscope feeds
- **Multi-Client Support**: Concurrent sessions for multiple users/applications  
- **RTX A5000 Optimized**: Memory-efficient setup for 24GB VRAM
- **Windows Host Support**: Complete Windows server deployment
- **Python SDK**: Easy programmatic access for client applications
- **Low Latency**: Optimized for real-time microscope control workflows

## System Requirements

### Host Server (Windows)
- **GPU**: NVIDIA RTX A5000 (24GB VRAM) or equivalent
- **RAM**: 32GB+ system memory recommended
- **OS**: Windows 10/11 with WSL2 or native Windows
- **CUDA**: Version 11.8 or newer
- **Storage**: 50GB+ free space for models

### Client Systems
- **Python**: 3.8+ with pip
- **Network**: Local network access to host server
- **Libraries**: opencv-python, numpy, requests

## Installation

### Step 1: Host Server Setup (Windows)

#### Download and Install Ollama

1. **Download Ollama for Windows**:
   ```bash
   # Visit https://ollama.com/download/windows
   # Download the Windows installer (ollama-windows-amd64.exe)
   ```

2. **Install Ollama**:
   - Run the downloaded installer as Administrator
   - Follow the installation wizard
   - Ollama will be installed to `C:\Users\{username}\AppData\Local\Programs\Ollama`

3. **Verify Installation**:
   ```cmd
   ollama --version
   ```

#### Configure GPU Support

1. **Install NVIDIA Drivers**:
   - Download latest drivers for RTX A5000 from NVIDIA website
   - Restart after installation

2. **Verify CUDA**:
   ```cmd
   nvidia-smi
   ```
   Should show your RTX A5000 with 24GB memory

#### Download Llama 3.2 Vision Model

1. **Pull the Model**:
   ```cmd
   # This downloads the 11B Vision model (~7GB with quantization)
   ollama pull llama3.2-vision:11b
   ```

2. **Alternative Smaller Model** (if memory issues):
   ```cmd
   # Lighter alternative
   ollama pull llava:13b
   ```

#### Start the Server

1. **Configure Concurrent Access**:
   ```cmd
   # Set environment variables for optimal RTX A5000 performance
   set OLLAMA_MAX_LOADED_MODELS=1
   set OLLAMA_NUM_PARALLEL=8
   set OLLAMA_MAX_QUEUE=64
   ```

2. **Start Ollama Server**:
   ```cmd
   ollama serve
   ```
   
   Server will start on `http://localhost:11434`

3. **Test the Server**:
   ```cmd
   # In a new terminal
   ollama run llama3.2-vision:11b "Describe this image" --image path/to/test-image.jpg
   ```

### Step 2: Client Setup

#### Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install client libraries
pip install ollama opencv-python numpy pillow requests
```

#### Basic Client Connection Test

```python
# test_connection.py
import ollama

# Connect to the server
client = ollama.Client(host='http://your-server-ip:11434')

# Test connection
try:
    response = client.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': 'Hello, can you see images?'
        }]
    )
    print("Connection successful!")
    print(response['message']['content'])
except Exception as e:
    print(f"Connection failed: {e}")
```

## Usage Examples

### Basic Image Analysis

```python
import ollama
import base64

# Initialize client
client = ollama.Client(host='http://your-server-ip:11434')

# Load and encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Analyze microscope image
def analyze_microscope_image(image_path, prompt):
    base64_image = encode_image(image_path)
    
    response = client.chat(
        model='llama3.2-vision:11b',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [base64_image]
        }]
    )
    return response['message']['content']

# Example usage
result = analyze_microscope_image(
    "microscope_sample.jpg",
    "Analyze this microscope image. Identify any cellular structures and suggest optimal focus adjustments."
)
print(result)
```

### Real-time Microscope Integration

```python
import ollama
import cv2
import base64
import asyncio
from io import BytesIO
import numpy as np

class MicroscopeVLM:
    def __init__(self, server_host='http://localhost:11434'):
        self.client = ollama.AsyncClient(host=server_host)
        self.model = 'llama3.2-vision:11b'
    
    async def analyze_frame(self, frame, prompt):
        # Convert OpenCV frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = await self.client.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }]
        )
        return response['message']['content']
    
    async def continuous_analysis(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 30th frame to avoid overwhelming the server
            if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to analyze
                analysis = await self.analyze_frame(
                    frame, 
                    "Analyze this microscope view. Suggest focus and lighting adjustments."
                )
                print(f"Analysis: {analysis}")
            
            cv2.imshow('Microscope Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage
async def main():
    microscope = MicroscopeVLM()
    await microscope.continuous_analysis()

# Run the async function
asyncio.run(main())
```

### Multi-Session Client Management

```python
import ollama
import asyncio
import uuid

class SessionManager:
    def __init__(self, server_host='http://localhost:11434'):
        self.client = ollama.AsyncClient(host=server_host)
        self.model = 'llama3.2-vision:11b'
        self.sessions = {}
    
    def create_session(self, user_id):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'user_id': user_id,
            'conversation_history': []
        }
        return session_id
    
    async def process_image(self, session_id, image_data, prompt):
        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")
        
        session = self.sessions[session_id]
        
        # Add to conversation history
        session['conversation_history'].append({
            'role': 'user',
            'content': prompt,
            'images': [image_data] if image_data else None
        })
        
        response = await self.client.chat(
            model=self.model,
            messages=session['conversation_history'][-5:]  # Keep last 5 messages
        )
        
        result = response['message']['content']
        session['conversation_history'].append({
            'role': 'assistant',
            'content': result
        })
        
        return result

# Example usage
async def demo_sessions():
    manager = SessionManager()
    
    # Create multiple sessions
    session1 = manager.create_session("researcher_1")
    session2 = manager.create_session("researcher_2")
    
    # Process images concurrently
    tasks = [
        manager.process_image(session1, None, "What should I look for in blood samples?"),
        manager.process_image(session2, None, "How do I adjust microscope lighting?")
    ]
    
    results = await asyncio.gather(*tasks)
    print("Session 1:", results[0])
    print("Session 2:", results[1])

asyncio.run(demo_sessions())
```

## Configuration

### Memory Optimization for RTX A5000

Create `ollama.conf` in your Ollama directory:

```ini
# Optimal settings for RTX A5000 (24GB VRAM)
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=8
OLLAMA_MAX_QUEUE=64
OLLAMA_GPU_MEMORY_FRACTION=0.9
OLLAMA_FLASH_ATTENTION=1
```

### Server Configuration

For production deployment, consider:

```yaml
# docker-compose.yml (optional)
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
    environment:
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_NUM_PARALLEL=8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Troubleshooting

### Common Windows Issues

1. **"CUDA not found" Error**:
   ```bash
   # Verify CUDA installation
   nvcc --version
   nvidia-smi
   
   # If missing, install CUDA Toolkit 11.8+
   ```

2. **Out of Memory Errors**:
   ```bash
   # Reduce parallel requests
   set OLLAMA_NUM_PARALLEL=4
   
   # Or use smaller model
   ollama pull llava:7b
   ```

3. **Slow Performance**:
   - Ensure GPU drivers are updated
   - Check that Ollama is using GPU: `nvidia-smi` should show GPU usage
   - Reduce image resolution before sending to model

4. **Connection Refused**:
   - Check Windows Firewall settings
   - Ensure Ollama server is running: `netstat -an | findstr 11434`
   - Verify client is connecting to correct IP address

### RTX A5000 Specific

- **Memory Usage**: Monitor with `nvidia-smi` - should stay under 20GB
- **Temperature**: Ensure adequate cooling under continuous load
- **Power**: RTX A5000 draws up to 230W - ensure adequate PSU

## Performance Benchmarks

Expected performance on RTX A5000:

- **Model Loading**: ~10-15 seconds for Llama 3.2-11B Vision
- **Inference Speed**: 15-25 tokens/second for text, 3-8 seconds for image analysis
- **Concurrent Sessions**: 4-8 simultaneous clients (depending on request complexity)
- **Memory Usage**: ~12-16GB VRAM with quantized model

## API Reference

### REST API Endpoints

```bash
# Generate completion
POST /api/generate
Content-Type: application/json

{
  "model": "llama3.2-vision:11b",
  "prompt": "Analyze this microscope image",
  "images": ["base64_encoded_image"],
  "stream": false
}

# Chat completion
POST /api/chat
Content-Type: application/json

{
  "model": "llama3.2-vision:11b",
  "messages": [
    {
      "role": "user",
      "content": "What do you see in this image?",
      "images": ["base64_encoded_image"]
    }
  ]
}
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Create Pull Request

## License

MIT License - see LICENSE file for details.

## Support

For issues specific to:
- **Ollama**: https://github.com/ollama/ollama/issues
- **Llama Models**: https://github.com/meta-llama/llama-models
- **This Project**: Create an issue in this repository