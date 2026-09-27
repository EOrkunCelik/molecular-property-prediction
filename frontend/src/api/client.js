const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new ApiError(detail, response.status, detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  health: () => request("/health"),

  predict: (smiles) =>
    request("/predict", {
      method: "POST",
      body: JSON.stringify({ smiles }),
    }),

  listPredictions: (limit = 20, offset = 0) =>
    request(`/predictions?limit=${limit}&offset=${offset}`),

  getPrediction: (id) => request(`/predictions/${id}`),

  structureImageUrl: (id) => `${API_BASE_URL}/predictions/${id}/structure`,

  visualizeSmilesUrl: (smiles) =>
    `${API_BASE_URL}/visualize?smiles=${encodeURIComponent(smiles)}`,
};

export { ApiError };
