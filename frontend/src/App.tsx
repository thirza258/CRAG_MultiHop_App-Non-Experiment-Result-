import { Routes, Route } from "react-router-dom";
import {
  ChatLayout,
  LandingPageLayout,
  LoginPageLayout,
} from "./layout/_layout";

import Chatbot from "./pages/Chatbot";
import LoginPage from "./pages/LoginPage";
import ErrorPage from "./pages/ErrorPage";
import Docs from "./pages/Docs";
import Landing from "./pages/Landing";
import About from "./pages/About";

function App() {
  return (
      <Routes>

        {/* Public, indexable pages */}
        <Route element={<LandingPageLayout />}>
          <Route path="/" element={<Landing />} />
          <Route path="/docs" element={<Docs />} />
          <Route path="/about" element={<About />} />
          <Route path="/error" element={<ErrorPage />} />
          {/* Nginx serves index.html for unknown paths, so catch them here
              instead of rendering a blank page. */}
          <Route path="*" element={<ErrorPage />} />
        </Route>

        {/* The app itself */}
        <Route element={<ChatLayout />}>
          <Route path="/chat" element={<Chatbot />} />
        </Route>

        <Route element={<LoginPageLayout />}>
          <Route path="/login" element={<LoginPage />} />
        </Route>

      </Routes>
  );
}

export default App;
