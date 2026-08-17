import React, { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { Menu, X, User, LogOut, Settings } from "lucide-react";
import { Button } from "../components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "../components/ui/avatar";

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    const storedUser = localStorage.getItem("username");
    if (storedUser) {
      setUsername(storedUser);
      setEmail(localStorage.getItem("email") || null);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("username");
    localStorage.removeItem("token");
    setUsername(null);
    navigate("/login");
  };

  const getInitials = (name: string) => {
    return name.substring(0, 2).toUpperCase();
  };

  const navLinks = [
    { path: "/", label: "Home" },
    { path: "/chat", label: "Chat" },
    { path: "/docs", label: "Docs" },
    { path: "/about", label: "About" },
  ];

  return (
    <nav className="fixed top-0 w-full z-50 bg-slate-950/85 backdrop-blur-md border-b border-slate-800/80">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* Logo Section */}
        <Link to="/" className="flex items-center gap-2.5 group" aria-label="CRAG MultiHop RAG — home">
          <img src="/logo.svg" alt="CRAG Logo" className="h-7 w-7 transition-transform group-hover:scale-105" />
          <span className="whitespace-nowrap text-lg sm:text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-cyan-400">
            <span className="md:hidden">CRAG MultiHop RAG</span>
            <span className="hidden md:inline">
              CRAG MultiHop RAG
            </span>
          </span>
        </Link>

        {/* Desktop Navigation Links */}
        <div className="hidden md:flex items-center gap-7 text-sm font-medium">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                className={`transition-colors py-1 ${
                  isActive
                    ? "text-cyan-400 font-semibold border-b-2 border-cyan-400"
                    : "text-slate-400 hover:text-cyan-400"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>

        {/* User Account / Action CTA */}
        <div className="hidden md:flex items-center gap-4">
          {username ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-10 w-10 rounded-full focus:ring-2 focus:ring-cyan-500">
                  <Avatar className="h-10 w-10 border border-slate-700">
                    <AvatarImage src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${username}`} alt={username} />
                    <AvatarFallback className="bg-slate-800 text-cyan-400 font-bold">
                      {getInitials(username)}
                    </AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56 bg-slate-900 border-slate-800 text-slate-200" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none text-white">{username}</p>
                    <p className="text-xs leading-none text-slate-400">{email || `${username}@user.local`}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-slate-800" />
                <DropdownMenuItem className="focus:bg-slate-800 focus:text-cyan-400 cursor-pointer" onClick={() => navigate("/chat")}>
                  <User className="mr-2 h-4 w-4" />
                  <span>Open Chat</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="focus:bg-slate-800 focus:text-cyan-400 cursor-pointer" onClick={() => navigate("/docs")}>
                  <Settings className="mr-2 h-4 w-4" />
                  <span>Documentation</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-slate-800" />
                <DropdownMenuItem 
                  className="focus:bg-red-900/50 focus:text-red-400 text-red-400 cursor-pointer"
                  onClick={handleLogout}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button onClick={() => navigate("/login")} className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium shadow-md shadow-cyan-950/50">
              Get Started
            </Button>
          )}
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden text-slate-300 hover:text-white hover:bg-slate-800 p-2 rounded-md transition-colors"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
          aria-expanded={isMenuOpen}
        >
          {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile Menu Dropdown */}
      {isMenuOpen && (
        <div className="md:hidden bg-slate-950/95 border-b border-slate-800 p-5 flex flex-col gap-4 animate-in slide-in-from-top-5">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setIsMenuOpen(false)}
                className={`text-base font-medium py-1.5 transition-colors ${
                  isActive ? "text-cyan-400 font-semibold" : "text-slate-300 hover:text-cyan-400"
                }`}
              >
                {link.label}
              </Link>
            );
          })}

          <div className="h-px bg-slate-800/80 my-1" />
          
          {username ? (
            <>
              <div className="flex items-center gap-3 px-2 py-2">
                <Avatar className="h-9 w-9">
                  <AvatarImage src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${username}`} />
                  <AvatarFallback className="bg-slate-800 text-cyan-400">{getInitials(username)}</AvatarFallback>
                </Avatar>
                <div>
                  <p className="text-slate-100 font-medium text-sm">{username}</p>
                  <p className="text-slate-400 text-xs">{email || `${username}@user.local`}</p>
                </div>
              </div>
              <Button variant="destructive" onClick={handleLogout} className="w-full justify-start mt-1">
                <LogOut className="mr-2 h-4 w-4" /> Log out
              </Button>
            </>
          ) : (
            <Button className="w-full bg-cyan-600 hover:bg-cyan-500 text-white" onClick={() => navigate("/login")}>
              Get Started
            </Button>
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;