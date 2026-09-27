import { useEffect, useState, useCallback } from "react";
import { api, ApiError } from "./api/client.js";
import SmilesForm from "./components/SmilesForm.jsx";
import PredictionResult from "./components/PredictionResult.jsx";
import HistoryTable from "./components/HistoryTable.jsx";

export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [health, setHealth] = useState(null);

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const data = await api.listPredictions(10, 0);
      setHistory(data.items);
    } catch {
      // Non-fatal: history is a convenience panel, not core functionality.
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable", model_loaded: false, database_connected: false }));
    refreshHistory();
  }, [refreshHistory]);

  async function handlePredict(smiles) {
    setLoading(true);
    setError(null);
    try {
      const prediction = await api.predict(smiles);
      setResult(prediction);
      refreshHistory();
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail || "The prediction request failed.");
      } else {
        setError("Could not reach the API. Is the backend running?");
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectHistoryItem(id) {
    setLoading(true);
    setError(null);
    try {
      const prediction = await api.getPrediction(id);
      setResult(prediction);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch {
      setError("Could not load that prediction.");
    } finally {
      setLoading(false);
    }
  }

  const showDegradedBanner =
    health && (health.status !== "ok" || health.model_loaded === false);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Molecular Property Prediction Platform</h1>
        <p className="subtitle">
          Predict aqueous solubility (LogS) of small organic molecules from a SMILES
          string, using RDKit + a scikit-learn model trained on the ESOL dataset.
        </p>
      </header>

      {showDegradedBanner && (
        <div className="banner banner-warning">
          {health.status === "unreachable"
            ? "Cannot reach the backend API. Make sure it is running (see the README)."
            : !health.model_loaded
              ? "The backend is running, but no trained model is loaded yet. Run `python scripts/train_model.py`."
              : "The backend reports a degraded status."}
        </div>
      )}

      <main>
        <section className="card">
          <SmilesForm onSubmit={handlePredict} loading={loading} />
          {error && <div className="banner banner-error">{error}</div>}
        </section>

        {result && (
          <section className="card">
            <PredictionResult result={result} />
          </section>
        )}

        <section className="card">
          <h2>Recent predictions</h2>
          <HistoryTable items={history} onSelect={handleSelectHistoryItem} loading={historyLoading} />
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Educational / experimental project. Predictions are not a substitute for
          laboratory measurement and should not be used for medical, regulatory, or
          drug-discovery decision-making.
        </p>
      </footer>
    </div>
  );
}
