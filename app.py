import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
from typing import List, Dict

# Load the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Debug print
print(f".env file exists: {os.path.exists(dotenv_path)}")
print(f"PINECONE_API in env: {'PINECONE_API' in os.environ}")

from modules.RAG import RAG

app = FastAPI(title="LLM Recommender System API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_response(self, websocket: WebSocket, message: str):
        await websocket.send_text(message)
        # Send end marker to signify completion
        await websocket.send_text("[END]")

manager = ConnectionManager()

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message: {data}")
            
            try:
                # Parse the incoming message
                request_data = json.loads(data)
                
                # Extract the user_id and prompt from the request
                prompt = request_data.get("prompt", "")
                
                if not prompt:
                    error_msg = {"error": "No prompt provided"}
                    await manager.send_response(websocket, json.dumps(error_msg))
                    continue
                
                # Process the request with the RAG model
                recommendation_agent = RAG(user_id='1')
                response = recommendation_agent.agent(prompt)
                
                print("RESPONSE ===>", response)
                print("KEYS ===>", response.keys())
                
                # Format the response to match the expected structure
                formatted_response = {
                    "output": response["output"],
                    # Include any other fields from the response that are needed by the frontend
                }
                
                # Send the response back to the client
                await manager.send_response(websocket, json.dumps(formatted_response))
                
            except json.JSONDecodeError:
                error_msg = {"error": "Invalid JSON format"}
                await manager.send_response(websocket, json.dumps(error_msg))
            except Exception as e:
                error_msg = {"error": f"Processing error: {str(e)}"}
                await manager.send_response(websocket, json.dumps(error_msg))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "LLM Recommender System is running"}

# Root endpoint with basic information
@app.get("/")
async def root():
    return {
        "message": "LLM Recommender System API",
        "description": "API for generating recommendations using LLM",
        "endpoints": {
            "websocket": "/ws/chat",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)