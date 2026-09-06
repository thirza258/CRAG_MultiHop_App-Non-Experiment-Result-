import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import type { ErrorState } from "../types/types";
import { useSeo } from "../lib/seo";


const ErrorPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useSeo({
    title: "Page not found | CRAG MultiHop RAG",
    description: "This page does not exist.",
    path: location.pathname,
    noindex: true,
  });

  const state = location.state as ErrorState;
  
  const status = state?.status || 404;
  const error = state?.error || "Page Not Found";
  const message = state?.message || "We couldn't find the page you were looking for.";

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[hsl(var(--background))] text-[hsl(var(--foreground))]">
      <div className="max-w-md rounded border border-[hsl(var(--border))] p-8 flex flex-col items-center">
        <div className="flex items-center space-x-3 mb-4">
          <span className="text-5xl font-semibold text-[hsl(var(--muted-foreground))]" data-numeric>
            {status}
          </span>
          <h1 className="text-2xl font-semibold">
            {error}
          </h1>
        </div>
        <p className="mb-2 text-center text-[hsl(var(--muted-foreground))]">
          {message}
        </p>
        <button
          className="mt-6 rounded bg-[hsl(var(--primary))] px-5 py-2 text-sm font-medium text-[hsl(var(--primary-foreground))] hover:opacity-90"
          onClick={() => navigate("/")}
        >
          Back to Home
        </button>
      </div>
    </div>
  );
};

export default ErrorPage;