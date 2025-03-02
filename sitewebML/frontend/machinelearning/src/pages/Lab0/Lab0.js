import React, { useState, useEffect } from "react";
import "./Lab0.css";

const Lab0 = () => {
  const [message, setMessage] = useState("");
  const [logs, setLogs] = useState([]);
  const [imageUrl, setImageUrl] = useState(""); // ✅ Stocke l'image

  const executeScript = async () => {
    try {
      setMessage("⏳ Entraînement en cours...");
      setLogs([]);
      setImageUrl(""); // ✅ Réinitialise l'image

      const response = await fetch("http://127.0.0.1:8000/execute", {
        method: "POST",
      });

      const data = await response.json();
      setMessage(data.message);

      // ✅ Attendre un peu avant de récupérer l'image (temps d'entraînement)
      setTimeout(fetchImage, 5000);
    } catch (error) {
      setMessage("❌ Erreur lors de l'exécution du script");
    }
  };

  // ✅ Fonction pour récupérer l'image des résultats
  const fetchImage = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/results");
      const data = await response.json();

      if (data.file) {
        setImageUrl("http://127.0.0.1:8000" + data.file); // ✅ Met à jour l'image
      }
    } catch (error) {
      console.error("Erreur lors de la récupération de l'image :", error);
    }
  };

  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8000/logs");

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLogs(data.logs);
    };

    return () => {
      socket.close();
    };
  }, []);

  return (
    <div className="lab0-container">
      <h1 className="lab0-title">Lab 0 - Q-Learning & FrozenLake</h1>

      <p className="lab0-intro">
        Ce laboratoire explore l'algorithme de{" "}
        <span className="lab0-highlight">Q-Learning</span> appliqué à
        l'environnement <span className="lab0-highlight">FrozenLake</span>.
      </p>

      <button className="lab0-button" onClick={executeScript}>
        ▶️ Lancer l'entraînement
      </button>

      {message && <p className="lab0-status">{message}</p>}

      <div className="lab0-logs">
        <h2 className="lab0-subtitle">📜 Logs en direct</h2>
        <div className="lab0-log-container">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <p key={index} className="lab0-log">
                {log}
              </p>
            ))
          ) : (
            <p className="lab0-log">En attente des logs...</p>
          )}
        </div>
      </div>

      {/* ✅ Affichage de l'image si elle est disponible */}
      {imageUrl && (
        <div className="lab0-image-container">
          <h2 className="lab0-subtitle">📊 Résultats de l'entraînement</h2>
          <img
            src={imageUrl}
            alt="Graphique des résultats"
            className="lab0-image"
          />
        </div>
      )}
    </div>
  );
};

export default Lab0;
