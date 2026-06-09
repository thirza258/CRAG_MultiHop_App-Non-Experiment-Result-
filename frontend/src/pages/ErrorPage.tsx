import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import type { ErrorState } from "../types/types";


const ErrorPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const state = location.state as ErrorState;
  
  const status = state?.status || 404;
  const error = state?.error || "Page Not Found";
  const message = state?.message || "We couldn't find the page you were looking for.";

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-950 text-white">
      <div className="bg-slate-900 rounded-lg shadow-lg p-8 border border-slate-800 flex flex-col items-center max-w-md">
        <div className="flex items-center space-x-3 mb-4">
          <span className="text-5xl font-bold text-red-500">
            {status}
          </span>
          <h1 className="text-3xl font-bold text-white">
            {error}
          </h1>
        </div>
        <p className="mb-2 text-slate-400 text-center">
          {message}
        </p>
        <button
          className="mt-6 px-6 py-2 rounded-md bg-cyan-600 text-white font-semibold hover:bg-cyan-700 transition-colors"
          onClick={() => navigate("/")}
        >
          Back to Home
        </button>
      </div>
    </div>
  );
};

export default ErrorPage;