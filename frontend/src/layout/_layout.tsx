import NavBar from "../components/Navbar";
import Sidebar from "../components/Sidebar";
import { Outlet } from "react-router-dom";

const ChatLayout = () => {
  
  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <NavBar /> 
      <div className="flex flex-1 overflow-hidden pt-16">
        <Sidebar />
        <main className="flex-1 overflow-y-auto bg-gray-100 relative">
           <Outlet />
        </main>
      </div>
    </div>
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

