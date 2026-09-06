import React, { useState } from "react";
import  service  from "../services/service";
import { useNavigate } from "react-router-dom";
import { useSeo } from "../lib/seo";

const LoginPage: React.FC = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const navigate = useNavigate();

  useSeo({
    title: "Sign in | CRAG MultiHop RAG",
    description:
      "Sign in with a username and email to start chatting with your documents.",
    path: "/login",
    noindex: true,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !email) {
      alert("Please enter both username and email.");
      return;
    }
    service.signUp(email, username)
      .then(response => {
        if (response.status !== 200 && response.status !== 201) {
          throw new Error(response.message);
        }
        const { username, email } = response.data;

        localStorage.setItem("username", username);
        localStorage.setItem("email", email);

        navigate("/chat");
      })
      .catch(error => {
        navigate("/error", {
          state: {
            status: error?.response?.status || 500,
            error: "Sign Up Failed",
            message: error?.response?.data?.message || error.message || "Sign up failed."
          }
        });
      });
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[hsl(var(--background))] px-4">
      <form
        className="w-80 space-y-4 rounded border border-[hsl(var(--border))] p-8"
        onSubmit={handleSubmit}
      >
        <h1 className="mb-1 text-center text-xl font-semibold">Sign in</h1>
        <p className="mb-6 text-center text-sm text-[hsl(var(--muted-foreground))]">CRAG MultiHop RAG — research build</p>
        <div className="my-4 h-px w-full bg-[hsl(var(--border))]" />
        
        <div>
          <label className="mb-2 block text-sm font-medium text-[hsl(var(--foreground))]" htmlFor="username">
            Username
          </label>
          <input
            id="username"
            type="text"
            className="w-full rounded border border-[hsl(var(--input))] bg-[hsl(var(--background))] p-2 text-sm text-[hsl(var(--foreground))] placeholder:text-[hsl(var(--muted-foreground))]"
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoComplete="username"
          />
        </div>
        <div>
          <label className="mb-2 block text-sm font-medium text-[hsl(var(--foreground))]" htmlFor="email">
            Email
          </label>
          <input
            id="email"
            type="email"
            className="w-full rounded border border-[hsl(var(--input))] bg-[hsl(var(--background))] p-2 text-sm text-[hsl(var(--foreground))] placeholder:text-[hsl(var(--muted-foreground))]"
            value={email}
            onChange={e => setEmail(e.target.value)}
            autoComplete="email"
          />
        </div>
        <button
          type="submit"
          className="w-full rounded bg-[hsl(var(--primary))] py-2 text-sm font-medium text-[hsl(var(--primary-foreground))] transition-opacity hover:opacity-90"
        >
          Login
        </button>
        <button
          type="button"
          onClick={() => navigate("/")}
          className="mt-2 w-full rounded border border-[hsl(var(--border))] py-2 text-sm font-medium text-[hsl(var(--foreground))] transition-colors hover:bg-[hsl(var(--muted))]"
        >
          Go Back
        </button>
      </form>
    </div>
  );
};

export default LoginPage;
