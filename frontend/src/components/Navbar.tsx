import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Menu, X, User, LogOut, Settings, CreditCard } from "lucide-react";
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

  // Helper to get initials
  const getInitials = (name: string) => {
    return name.substring(0, 2).toUpperCase();
  };

  return (
    <nav className="fixed top-0 w-full z-50 bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* Logo Section */}
        <Link to="/" className="flex items-center gap-2" aria-label="CRAG MultiHop RAG — home">
          {/* The full name doesn't fit a 64px bar on phones. */}
          <span className="whitespace-nowrap text-lg sm:text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
            <span className="md:hidden">CRAG MultiHop RAG</span>
            <span className="hidden md:inline">
              Corrective · Multihop · Reranker RAG
            </span>
          </span>
        </Link>

        {/* Desktop Menu */}
        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
          <Link to="/" className="hover:text-cyan-400 transition-colors">Home</Link>
          <Link to="/chat" className="hover:text-cyan-400 transition-colors">Chat</Link>
          <Link to="/docs" className="hover:text-cyan-400 transition-colors">Docs</Link>
          <Link to="/about" className="hover:text-cyan-400 transition-colors">About</Link>
        </div>


        <div className="hidden md:flex items-center gap-4">
          {username ? (

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="relative h-10 w-10 rounded-full">
                  <Avatar className="h-10 w-10 border border-slate-700">
                    <AvatarImage src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${username}`} alt={username} />
                    <AvatarFallback className="bg-slate-800 text-cyan-400">
                      {getInitials(username)}
                    </AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56 bg-slate-900 border-slate-800 text-slate-200" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none text-white">{username}</p>
                    <p className="text-xs leading-none text-slate-400">{email}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-slate-800" />
                <DropdownMenuItem className="focus:bg-slate-800 focus:text-cyan-400 cursor-pointer">
                  <User className="mr-2 h-4 w-4" />
                  <span>Profile</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="focus:bg-slate-800 focus:text-cyan-400 cursor-pointer">
                  <CreditCard className="mr-2 h-4 w-4" />
                  <span>Billing</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="focus:bg-slate-800 focus:text-cyan-400 cursor-pointer">
                  <Settings className="mr-2 h-4 w-4" />
                  <span>Settings</span>
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

            <>
              {/* Sign-in is the only entry point — there is no separate
                  registration flow, a username and email creates the account. */}
              <Button onClick={() => navigate("/login")} className="bg-cyan-600 hover:bg-cyan-700 text-white">
                Get Started
              </Button>
            </>
          )}
        </div>

        <button
          className="md:hidden text-white hover:bg-slate-800 p-2 rounded-md"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
          aria-expanded={isMenuOpen}
        >
          {isMenuOpen ? <X /> : <Menu />}
        </button>
      </div>


      {isMenuOpen && (
        <div className="md:hidden bg-slate-900 border-b border-slate-800 p-4 flex flex-col gap-4 animate-in slide-in-from-top-5">
          <Link to="/" onClick={() => setIsMenuOpen(false)} className="text-slate-300 hover:text-cyan-400">Home</Link>
          <Link to="/chat" onClick={() => setIsMenuOpen(false)} className="text-slate-300 hover:text-cyan-400">Chat</Link>
          <Link to="/docs" onClick={() => setIsMenuOpen(false)} className="text-slate-300 hover:text-cyan-400">Docs</Link>
          <Link to="/about" onClick={() => setIsMenuOpen(false)} className="text-slate-300 hover:text-cyan-400">About</Link>

          <div className="h-px bg-slate-800 my-2" />
          
          {username ? (
            <>
              <div className="flex items-center gap-3 px-2 py-2">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${username}`} />
                  <AvatarFallback>{getInitials(username)}</AvatarFallback>
                </Avatar>
                <span className="text-slate-200 font-medium">{username}</span>
              </div>
              <Button variant="destructive" onClick={handleLogout} className="w-full justify-start">
                <LogOut className="mr-2 h-4 w-4" /> Log out
              </Button>
            </>
          ) : (
            <Button className="w-full bg-cyan-600" onClick={() => navigate("/login")}>
              Get Started
            </Button>
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;