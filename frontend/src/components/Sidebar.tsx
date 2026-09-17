import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import service from "../services/service";
import { FileMetadata, ConversationItem } from "../interface";
import PipelineConfigPanel from "./settings/PipelineConfigPanel";
import ApiKeysPanel from "./settings/ApiKeysPanel";

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const [files, setFiles] = useState<FileMetadata[]>([]);
  const [history, setHistory] = useState<ConversationItem[]>([]);
  const usernameFromStorage = localStorage.getItem("username");

  const fetchFiles = async (username: string) => {
    try {
      const response = await service.getDocumentInfo(username);
      const actualFiles = response?.data || response;
      if (Array.isArray(actualFiles)) {
        setFiles(actualFiles);
      } else if (actualFiles && typeof actualFiles === 'object' && actualFiles.id) {
        setFiles([actualFiles]);
      } else {
        setFiles([]);
      }
    } catch (err) {
      console.error("Failed to fetch files:", err);
      setFiles([]);
    }
  };

  const handleDeleteFile = async (fileId: string, fileName: string) => {
    if (!usernameFromStorage) return;
    const confirmDelete = window.confirm(`Are you sure you want to delete "${fileName}"?`);
    if (!confirmDelete) return;

    try {
      await service.deleteDocument(fileId, usernameFromStorage);
      setFiles(prev => prev.filter(f => f.id !== fileId));
    } catch (error) {
      console.error("Delete failed:", error);
      alert("Failed to delete document. Please try again.");
    }
  };

  useEffect(() => {
    if (!usernameFromStorage) {
      navigate("/login");
      return;
    }

    const fetchData = async () => {
      try {
        const cacheKey = `chat_history_${usernameFromStorage}`;
        const cachedHistory = sessionStorage.getItem(cacheKey);

        const filesPromise = fetchFiles(usernameFromStorage);

        const historyPromise = cachedHistory
          ? Promise.resolve(JSON.parse(cachedHistory))
          : service.getConversationHistory(usernameFromStorage)
              .then((res) => {
                const unwrappedData = res?.data || res || [];
                const dataToCache = Array.isArray(unwrappedData) ? unwrappedData : [];
                sessionStorage.setItem(cacheKey, JSON.stringify(dataToCache));
                return dataToCache;
              })
              .catch(err => {
                console.error("Failed to fetch history:", err);
                return [];
              });

        const [historyResponse] = await Promise.all([historyPromise, filesPromise]);
        setHistory(Array.isArray(historyResponse) ? historyResponse : []);
      } catch (error) {
        console.error("Error fetching sidebar data:", error);
        setHistory([]);
        setFiles([]);
      }
    };

    fetchData();
  }, [usernameFromStorage, navigate]);

return (
  <div className="w-1/3 min-w-[300px] max-w-[400px] h-full flex flex-col border-r border-[hsl(var(--border))] bg-[hsl(var(--card))] text-[hsl(var(--card-foreground))] shadow-xl">
    {/* Header - fixed */}
    <div className="p-4 border-b border-[hsl(var(--border))] flex-shrink-0">
      <h2 className="text-xl font-bold tracking-tight text-[hsl(var(--foreground))]">
        Context Details
      </h2>
    </div>

    {/* Scrollable sections container */}
    <div className="flex-1 overflow-y-auto min-h-0 p-4 space-y-4">

      {/* Per-query pipeline configuration */}
      <div className="space-y-3">
        <PipelineConfigPanel />
        <ApiKeysPanel />
      </div>

      <hr className="border-[hsl(var(--border))]" />

      {/* Current Active Content Section */}
      <section>
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-[hsl(var(--primary))]">
            Document List
          </h3>
          <span className="text-xs text-[hsl(var(--muted-foreground))]">
            {files?.length || 0} Items
          </span>
        </div>
        <div>
          {!files?.length ? (
            <div className="p-4 rounded bg-[hsl(var(--muted))] border border-dashed border-[hsl(var(--border))] text-center">
              <p className="text-sm text-[hsl(var(--muted-foreground))]">
                No active content selected.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {files.map((file) => (
                <div
                  key={file.id}
                  className="group flex items-center justify-between p-3 rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] hover:bg-[hsl(var(--muted))] transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-medium truncate" title={file.name}>
                        {file.name}
                      </span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-[hsl(var(--primary))/10] text-[hsl(var(--primary))] font-mono">
                        {file.source_type || "Unknown"}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteFile(file.id, file.name)}
                    className="ml-2 text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--destructive))] transition-colors flex-shrink-0"
                    aria-label="Delete document"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M3 6h18" />
                      <path d="M8 6v-2a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                      <line x1="10" y1="11" x2="10" y2="17" />
                      <line x1="14" y1="11" x2="14" y2="17" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Divider */}
      <hr className="border-[hsl(var(--border))]" />

      {/* Recent History Section */}
      <section>
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-[hsl(var(--primary))]">
            Recent History
          </h3>
          <span className="text-xs text-[hsl(var(--muted-foreground))]">
            {history?.length || 0} Items
          </span>
        </div>
        <div>
          {!history?.length ? (
            <p className="text-sm text-[hsl(var(--muted-foreground))] italic">
              No recent history.
            </p>
          ) : (
            <div className="space-y-2">
              {history.map((item, index) => (
                <div
                  key={index}
                  className="group p-3 rounded-md border border-transparent hover:border-[hsl(var(--border))] hover:bg-[hsl(var(--muted))] cursor-pointer transition-all duration-200"
                >
                  <div className="flex justify-between items-start">
                    <p className="text-sm font-medium group-hover:text-[hsl(var(--primary))] transition-colors truncate pr-2" title={item.query}>
                      {item.query}
                    </p>
                    <span className="text-[10px] bg-[hsl(var(--muted))] border border-[hsl(var(--border))] px-1 rounded text-[hsl(var(--muted-foreground))] whitespace-nowrap">
                      Chat
                    </span>
                  </div>
                  <p className="text-xs text-[hsl(var(--muted-foreground))] mt-1 line-clamp-2" title={item.response}>
                    {item.response}
                  </p>
                  <p className="text-[10px] text-[hsl(var(--muted-foreground))] mt-2 opacity-70">
                    {item.created_at ? new Date(item.created_at).toLocaleString() : "Unknown"}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
      
    </div>
  </div>
);
};

export default Sidebar;