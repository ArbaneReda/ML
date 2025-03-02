from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import subprocess
import asyncio
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Autoriser les requêtes du frontend React
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API FastAPI 🚀"}

@app.post("/execute")
def execute():
    subprocess.Popen(["python", "train_test_qtable.py"])  # ✅ Exécute l'entraînement
    return {"message": "Entraînement lancé 🚀"}

@app.get("/results")
def get_results():
    if os.path.exists("results.png"):
        return {"message": "Résultats disponibles 🚀", "file": "results.png"}
    return {"message": "Pas encore de résultats."}

# ✅ WebSocket pour récupérer les logs en temps réel
@app.websocket("/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        last_size = 0
        while True:
            await asyncio.sleep(1)
            if os.path.exists("logs.txt"):
                with open("logs.txt", "r", encoding="utf-8") as f:
                    logs = f.readlines()
                
                if len(logs) > last_size:  # ✅ Envoie seulement les nouvelles lignes
                    new_logs = logs[last_size:]
                    await websocket.send_json({"logs": new_logs})
                    last_size = len(logs)  # ✅ Mise à jour du compteur

    except WebSocketDisconnect:
        print("INFO: WebSocket fermé")

    except asyncio.CancelledError:
        print("INFO: WebSocket arrêté proprement")  # ✅ Ignore les erreurs de WebSocket
