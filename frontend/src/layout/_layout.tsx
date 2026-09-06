import NavBar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import { Outlet } from "react-router-dom";
import { ApiKeysProvider } from "../context/ApiKeysContext";
import { PipelineConfigProvider } from "../context/PipelineConfigContext";

const ChatLayout = () => {

  return (
    // Both providers must sit above BOTH Sidebar (which edits the pipeline
    // config and the keys) and the Outlet (Chatbot, which sends them with each
    // query). They are separate contexts on purpose — see ApiKeysContext.
    <PipelineConfigProvider>
      <ApiKeysProvider>
        <div className="flex flex-col h-screen overflow-hidden">
          <NavBar />
          <div className="flex flex-1 overflow-hidden pt-16">
            <Sidebar />
            <main className="flex-1 overflow-y-auto bg-[hsl(var(--background))] relative">
              <Outlet />
            </main>
          </div>
        </div>
      </ApiKeysProvider>
    </PipelineConfigProvider>
  );
};

const LandingPageLayout = () => {
  return (
    <>
      <NavBar />
      <Outlet />
    </>
  );
};


const LoginPageLayout = () => {
  return (
    <>
      <Outlet />
    </>
  );
};


export { 
    ChatLayout, 
    LandingPageLayout,
    LoginPageLayout,
};

