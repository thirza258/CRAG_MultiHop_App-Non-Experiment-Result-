import React, { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Shield, FileText, Server, Cpu, ExternalLink, ArrowLeft } from "lucide-react";

export const Legal: React.FC = () => {
  const location = useLocation();
  const initialTab = location.pathname.includes("terms") ? "terms" : "privacy";
  const [activeTab, setActiveTab] = useState<"privacy" | "terms">(initialTab);

  return (
    <div className="min-h-screen bg-background text-foreground py-16 px-6 sm:px-8">
      <div className="mx-auto max-w-4xl">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm text-primary hover:underline mb-8"
        >
          <ArrowLeft className="h-4 w-4" /> Back to Home
        </Link>

        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-6 mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
              {activeTab === "privacy" ? (
                <Shield className="h-8 w-8 text-primary" />
              ) : (
                <FileText className="h-8 w-8 text-primary" />
              )}
              {activeTab === "privacy" ? "Privacy Policy" : "Terms of Service"}
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              CRAG Multi-Hop RAG Platform · Effective November 2026
            </p>
          </div>

          <div className="flex rounded-lg border border-border bg-muted p-1">
            <button
              type="button"
              onClick={() => setActiveTab("privacy")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                activeTab === "privacy"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Privacy Policy
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("terms")}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-all ${
                activeTab === "terms"
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Terms of Service
            </button>
          </div>
        </div>

        {activeTab === "privacy" ? (
          <div className="space-y-8 text-sm leading-relaxed text-muted-foreground">
            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-foreground mb-3">
                <Server className="h-5 w-5 text-primary" />
                1. 100% Self-Hosted & Local Data Isolation
              </h2>
              <p className="mb-3">
                CRAG Multi-Hop is an open-source, fully self-hosted corrective retrieval-augmented generation engine.
                All uploaded PDF documents, extracted text chunks, vector embeddings stored in ChromaDB,
                session logs, and evaluation metrics reside exclusively on your self-hosted host server.
              </p>
              <p>
                The platform contains no telemetry, tracking beacons, cross-site analytics, or data monetization pipelines.
                Your research corpus never leaves your self-hosted database boundaries without explicit user action.
              </p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold flex items-center gap-2 text-foreground mb-3">
                <Cpu className="h-5 w-5 text-primary" />
                2. AI Inference & OpenRouter Provider Disclosure
              </h2>
              <p className="mb-3">
                When generating query decompositions, performing multi-hop reasoning, corrective source evaluation,
                or final answer synthesis, the backend interacts with large language models through <strong>OpenRouter</strong>.
              </p>
              <p className="mb-3">
                Under this architecture:
              </p>
              <ul className="list-disc list-inside space-y-1 pl-2 mb-4">
                <li>Only the prompt and minimal retrieved context chunks necessary to answer the query are transmitted over TLS to OpenRouter&apos;s API endpoints.</li>
                <li>Your entire private document database is not synchronized or shared with external model hosts.</li>
              </ul>
              <div className="rounded-lg border border-border bg-muted/60 p-4 space-y-2">
                <p className="text-xs font-semibold uppercase tracking-wider text-foreground">
                  Upstream AI Provider Policies
                </p>
                <div className="flex flex-wrap gap-4 text-xs">
                  <a
                    href="https://openrouter.ai/privacy"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-primary hover:underline"
                  >
                    OpenRouter Privacy Policy <ExternalLink className="h-3 w-3" />
                  </a>
                  <a
                    href="https://openrouter.ai/terms"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-primary hover:underline"
                  >
                    OpenRouter Terms of Service <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              </div>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">3. Document &amp; Cache Management</h2>
              <p>
                Document uploads and intermediate vector representations are stored in containerized local volumes.
                Administrators can clear the vector database or flush indices at any time via project commands.
              </p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">4. Privacy Contact</h2>
              <p>
                For questions regarding data privacy or server operations, reach us at:{" "}
                <a href="mailto:nevatalapp@gmail.com" className="text-primary hover:underline font-mono">
                  nevatalapp@gmail.com
                </a>
              </p>
            </section>
          </div>
        ) : (
          <div className="space-y-8 text-sm leading-relaxed text-muted-foreground">
            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">1. Terms Acceptance</h2>
              <p>
                By utilizing this self-hosted instance of CRAG Multi-Hop, you agree to comply with these Terms of Service.
                This software is intended for research, document evaluation, and technical exploration.
              </p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">2. AI Output &amp; Academic Notice</h2>
              <p className="mb-2">
                Answers, citations, and evaluation metrics generated by the multi-hop pipeline are powered by artificial intelligence
                and must be verified against source literature.
              </p>
              <p>
                The maintainers make no claims regarding zero-hallucination guarantees or scientific infallibility.
                Always cross-reference provided chunk citations with original source manuscripts.
              </p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">3. Acceptable Use</h2>
              <p>
                You agree not to utilize this pipeline to process confidential personal health information without compliance controls,
                nor submit queries designed to violate upstream model provider policies as outlined in OpenRouter&apos;s Terms of Service.
              </p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">4. Disclaimer of Warranty</h2>
              <p>
                The application is provided &ldquo;AS IS&rdquo; without warranties of any kind.
                In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
              </p>
            </section>

            <section className="rounded-xl border border-border bg-card p-6 text-card-foreground">
              <h2 className="text-lg font-semibold text-foreground mb-3">5. Terms Contact</h2>
              <p>
                For inquiries regarding usage terms, contact:{" "}
                <a href="mailto:nevatalapp@gmail.com" className="text-primary hover:underline font-mono">
                  nevatalapp@gmail.com
                </a>
              </p>
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

export default Legal;
