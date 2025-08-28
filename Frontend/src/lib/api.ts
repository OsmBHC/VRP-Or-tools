const API_BASE = "/api";

export type MatrixResponse = {
  success: boolean;
  message: string;
  execution_time: number;
  num_points: number;
  up_to_date?: boolean;
};

export async function uploadDataset(file: File): Promise<{ success: boolean; message: string; rows?: number }>{
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`/api/upload-dataset`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function generateMatrices(depot_lat: number, depot_lng: number): Promise<MatrixResponse> {
  const res = await fetch(`${API_BASE}/generate-matrices`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ depot_lat, depot_lng }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export type VRPParams = {
  num_vehicles: number;
  vehicle_capacity: number; // kg
  service_time: number; // minutes par client
  time_limit: number; // secondes
  vehicle_time_limit: number; // minutes par véhicule
  interval_seconds: number; // secondes entre les logs de solutions intermédiaires
};

export type VRPResponse = {
  success: boolean;
  message: string;
  vehicles_used: number;
  total_distance: number;
  total_load: number;
  routes: any[];
  execution_time: number;
};

export async function solveVRP(params: VRPParams): Promise<VRPResponse> {
  const res = await fetch(`${API_BASE}/solve-vrp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getStatus() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2000);
  try {
    const res = await fetch(`${API_BASE}/status`, { signal: controller.signal });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

export async function getSummary() {
  const res = await fetch(`${API_BASE}/summary`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getLastSolution() {
  const res = await fetch(`${API_BASE}/last-solution`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}
