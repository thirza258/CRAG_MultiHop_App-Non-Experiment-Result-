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

function App() {
  return (
      <Routes>


        <Route element={<ChatLayout />}>
          <Route path="/" element={<Chatbot />} />
        </Route>

        <Route element={<LoginPageLayout />}>
          <Route path="/login" element={<LoginPage />} />
        </Route>

        <Route element={<LandingPageLayout />}>
          <Route path="/error" element={<ErrorPage />} />
        </Route>

        <Route element={<LandingPageLayout />}>
          <Route path="/docs" element={<Docs />} />
        </Route>

      </Routes>
  );
}

export default App;
