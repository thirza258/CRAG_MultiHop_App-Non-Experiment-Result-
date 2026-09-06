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
    <nav className="fixed top-0 z-50 w-full border-b border-[hsl(var(--border))] bg-[hsl(var(--background))]">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* Logo Section */}
        <Link to="/" className="flex items-center gap-2.5" aria-label="CRAG MultiHop RAG — home">
          <img src="/logo.svg" alt="" className="h-6 w-6" />
          <span className="whitespace-nowrap text-base font-semibold text-[hsl(var(--foreground))]">
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
                    ? "text-[hsl(var(--foreground))] font-semibold border-b-2 border-[hsl(var(--primary))]"
                    : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]"
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
                <Button variant="ghost" className="relative h-9 w-9 rounded-full">
                  <Avatar className="h-9 w-9 border border-[hsl(var(--border))]">
                    <AvatarImage src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${username}`} alt={username} />
                    <AvatarFallback className="bg-[hsl(var(--muted))] text-xs font-medium text-[hsl(var(--foreground))]">
                      {getInitials(username)}
                    </AvatarFallback>
                  </Avatar>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-56 border-[hsl(var(--border))] bg-[hsl(var(--popover))] text-[hsl(var(--popover-foreground))]" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{username}</p>
                    <p className="text-xs leading-none text-[hsl(var(--muted-foreground))]">{email || `${username}@user.local`}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator className="bg-[hsl(var(--border))]" />
                <DropdownMenuItem className="cursor-pointer focus:bg-[hsl(var(--accent))]" onClick={() => navigate("/chat")}>
                  <User className="mr-2 h-4 w-4" />
                  <span>Open Chat</span>
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer focus:bg-[hsl(var(--accent))]" onClick={() => navigate("/docs")}>
                  <Settings className="mr-2 h-4 w-4" />
                  <span>Documentation</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator className="bg-[hsl(var(--border))]" />
                <DropdownMenuItem 
                  className="cursor-pointer text-[hsl(var(--destructive))] focus:bg-[hsl(var(--accent))] focus:text-[hsl(var(--destructive))]"
                  onClick={handleLogout}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Log out</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button onClick={() => navigate("/login")} className="bg-[hsl(var(--primary))] font-medium text-[hsl(var(--primary-foreground))] hover:opacity-90">
              Get Started
            </Button>
          )}
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="rounded p-2 text-[hsl(var(--muted-foreground))] transition-colors hover:text-[hsl(var(--foreground))] md:hidden"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
          aria-expanded={isMenuOpen}
        >
          {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile Menu Dropdown */}
      {isMenuOpen && (
        <div className="flex flex-col gap-4 border-b border-[hsl(var(--border))] bg-[hsl(var(--background))] p-5 md:hidden">
          {navLinks.map((link) => {
            const isActive = location.pathname === link.path;
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setIsMenuOpen(false)}
                className={`text-base font-medium py-1.5 transition-colors ${
                  isActive ? "font-semibold text-[hsl(var(--foreground))]" : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--foreground))]"
                }`}
              >
                {link.label}
              </Link>
            );
          })}

          <div className="my-1 h-px bg-[hsl(var(--border))]" />
          
          {username ? (
            <>
              <div className="flex items-center gap-3 px-2 py-2">
                <Avatar className="h-9 w-9">
                  <AvatarImage src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${username}`} />
                  <AvatarFallback className="bg-[hsl(var(--muted))] text-xs font-medium">{getInitials(username)}</AvatarFallback>
                </Avatar>
                <div>
                  <p className="text-sm font-medium">{username}</p>
                  <p className="text-xs text-[hsl(var(--muted-foreground))]">{email || `${username}@user.local`}</p>
                </div>
              </div>
              <Button variant="destructive" onClick={handleLogout} className="w-full justify-start mt-1">
                <LogOut className="mr-2 h-4 w-4" /> Log out
              </Button>
            </>
          ) : (
            <Button className="w-full bg-[hsl(var(--primary))] text-[hsl(var(--primary-foreground))] hover:opacity-90" onClick={() => navigate("/login")}>
              Get Started
            </Button>
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;