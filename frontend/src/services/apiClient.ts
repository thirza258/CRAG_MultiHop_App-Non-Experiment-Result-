import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "/api";
const API_VERSION = import.meta.env.VITE_API_VERSION || "v1";

export const apiClient = axios.create({
    baseURL: `${API_BASE}/${API_VERSION}`,
    timeout: 30000,
});

