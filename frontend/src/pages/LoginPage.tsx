import React, { useState } from "react";
import  service  from "../services/service";
import { useNavigate } from "react-router-dom";

const LoginPage: React.FC = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const navigate = useNavigate();

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

        navigate("/");
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
    <div className="min-h-screen flex items-center justify-center bg-slate-950">
      <form
        className="bg-slate-900 p-8 rounded-lg shadow-md w-80 space-y-4 border border-slate-800"
        onSubmit={handleSubmit}
      >
        <h1 className="text-2xl font-bold mb-6 text-center text-white">Welcome Back!</h1>
        <h1 className="text-sm mb-6 text-center text-slate-400">RagReader alpha ver.</h1>
        <div className="w-full h-px bg-slate-700 my-4"></div>
        <h2 className="text-xl font-bold mb-4 text-center text-white">Login</h2>
        <div>
          <label className="block mb-2 text-sm font-medium text-slate-300" htmlFor="username">
            Username
          </label>
          <input
            id="username"
            type="text"
            className="w-full border border-slate-700 rounded p-2 bg-slate-800 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoComplete="username"
          />
        </div>
        <div>
          <label className="block mb-2 text-sm font-medium text-slate-300" htmlFor="email">
            Email
          </label>
          <input
            id="email"
            type="email"
            className="w-full border border-slate-700 rounded p-2 bg-slate-800 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={email}
            onChange={e => setEmail(e.target.value)}
            autoComplete="email"
          />
        </div>
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-500 transition-colors"
        >
          Login
        </button>
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="w-full bg-slate-700 text-white py-2 rounded hover:bg-slate-600 transition-colors mt-2"
        >
          Go Back
        </button>
      </form>
    </div>
  );
};

export default LoginPage;
