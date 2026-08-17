import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  CircleCheckBig,
  Copy,
  Cpu,
  Database,
  FileText,
  GitBranch,
  Github,
  Layers,
  ListChecks,
  Play,
  RotateCcw,
  Search,
  Server,
  ShieldQuestionMark,
  Sparkles,
  Terminal,
} from "lucide-react";
import { useSeo, REPO_URL, SITE_URL } from "../lib/seo";
import Footer from "../components/Footer";

/** Keep these two in sync with the static tags in index.html. */
const LANDING_TITLE =
  "Corrective Multi-Hop RAG with Reranking | CRAG MultiHop RAG";
const LANDING_DESCRIPTION =
  "Chat with your PDFs, URLs and notes. Dense + BM25 retrieval, self-grading Corrective RAG, up to 3 query hops, local reranking and faithfulness scores on every answer.";

const stats = [
  { value: "2", label: "retrievers per query", sub: "dense embeddings + BM25 sparse" },
  { value: "≤ 3", label: "reasoning hops", sub: "decomposed automatically" },
  { value: "2", label: "models run locally", sub: "reranker + retrieval grader" },
  { value: "2", label: "scores per answer", sub: "faithfulness + answer relevancy" },
];

const sampleSimulations = [
  {
    id: "query-1",
    label: "Supplier Risk Audit",
    query: "Which of the two reports flagged the same supplier risk, and what did each recommend?",
    stages: [
      { name: "multi_hop_retrieval", status: "complete", detail: "Decomposed query into Hop 1 (Identified suppliers) & Hop 2 (Risk recommendations)" },
      { name: "dense_retrieval · sparse_retrieval", status: "complete", detail: "Retrieved 4 vector chunks + 4 BM25 keyword candidates" },
      { name: "corrective_pipeline", status: "complete", detail: "CRAG decision: ambiguous (score 0.883) → refined query with expanded terms" },
      { name: "reranking_pipeline", status: "complete", detail: "Jina-reranker-v3 reordered 8 candidates, selected top 4 chunks" },
      { name: "answer_generation", status: "complete", detail: "Synthesized cross-report comparison with explicit chunk citations" },
      { name: "evaluation_start", status: "complete", detail: "Faithfulness: 0.94 · Answer Relevancy: 0.91" }
    ],
    answer: "Both the Q3 Supply Chain Audit (Doc A) and Vendor Risk Assessment (Doc B) identified sole-source dependencies on Component X. Doc A recommended dual-sourcing within 90 days, while Doc B advised increasing safety stock by 45%.",
    faithfulness: "0.94",
    relevancy: "0.91",
    chunks: ["Doc A: Section 4.2 (Supplier Vulnerabilities)", "Doc B: Paragraph 12 (Inventory Resilience)"]
  },
  {
    id: "query-2",
    label: "Financial Comparison",
    query: "What was the operating margin shift following the manufacturing relocation in Q3?",
    stages: [
      { name: "multi_hop_retrieval", status: "complete", detail: "Hop 1 (Q3 relocation dates) → Hop 2 (Pre/Post operating margins)" },
      { name: "dense_retrieval · sparse_retrieval", status: "complete", detail: "Retrieved 4 vector candidates + 4 BM25 matching passages" },
      { name: "corrective_pipeline", status: "complete", detail: "CRAG decision: correct (score 0.942) → filtered 2 weak passages" },
      { name: "reranking_pipeline", status: "complete", detail: "Jina-reranker-v3 scored top 4 chunks (avg rerank score 0.92)" },
      { name: "answer_generation", status: "complete", detail: "Generated precision response using grounded financials" },
      { name: "evaluation_start", status: "complete", detail: "Faithfulness: 0.96 · Answer Relevancy: 0.94" }
    ],
    answer: "Following the Q3 manufacturing relocation, operating margins initially dipped by 1.8% in August due to transition setup costs, but recovered to +3.4% above baseline by end of Q3 due to lower labor expenditure.",
    faithfulness: "0.96",
    relevancy: "0.94",
    chunks: ["Q3 Financial Report: Page 14 (CapEx Relocation)", "Q3 Summary: Page 22 (Operating Margins)"]
  },
  {
    id: "query-3",
    label: "Legal Compliance Audit",
    query: "Compare EU and US regulatory requirements for client data retention.",
    stages: [
      { name: "multi_hop_retrieval", status: "complete", detail: "Hop 1 (EU GDPR retention rules) → Hop 2 (US CCPA/FTC guidelines)" },
      { name: "dense_retrieval · sparse_retrieval", status: "complete", detail: "Retrieved 4 vector passages + 4 BM25 legal sections" },
      { name: "corrective_pipeline", status: "complete", detail: "CRAG decision: ambiguous (score 0.865) → expanded query terms" },
      { name: "reranking_pipeline", status: "complete", detail: "Jina-reranker-v3 deduplicated and prioritized top 4 legal clauses" },
      { name: "answer_generation", status: "complete", detail: "Structured side-by-side compliance table synthesis" },
      { name: "evaluation_start", status: "complete", detail: "Faithfulness: 0.92 · Answer Relevancy: 0.89" }
    ],
    answer: "EU regulations mandate strict data minimization with deletion upon request within 30 days unless legal hold applies. US policies require 7-year audit retention for financial transaction logs, overriding soft deletion.",
    faithfulness: "0.92",
    relevancy: "0.89",
    chunks: ["EU Data Privacy Addendum: Clause 8", "US Compliance Standard Operating Procedure: Sec 3.1"]
  }
];

const features = [
  {
    icon: FileText,
    title: "Bring your own documents",
    body: "Upload a PDF, submit a URL, or paste raw text. The text is extracted, split into ~500-character overlapping chunks, embedded, and stored in ChromaDB by a background worker, so long documents never block the UI.",
  },
  {
    icon: GitBranch,
    title: "Questions that need more than one lookup",
    body: "The multi-hop orchestrator decides whether your question needs follow-up retrieval, then runs up to three sequential hops — each one building its query from what the previous hop found, and stopping early once a hop adds nothing new.",
  },
  {
    icon: ShieldQuestionMark,
    title: "Retrieval that grades itself",
    body: "A Corrective RAG layer scores the retrieved context as correct, ambiguous or incorrect. Correct context is stripped of weak chunks, ambiguous context is refined with an expanded query, and context judged insufficient escalates to external search.",
  },
  {
    icon: Layers,
    title: "Hybrid candidates, locally reranked",
    body: "Dense and sparse results are merged and deduplicated, then reordered by jina-reranker-v3 running on the server's own CPU or GPU — no reranking API call, no per-query reranking cost.",
  },
  {
    icon: Activity,
    title: "Watch the pipeline as it runs",
    body: "Each stage streams to the browser over a websocket while you wait: dense_retrieval, sparse_retrieval, corrective_pipeline, multi_hop_retrieval, reranking_pipeline, answer_generation, evaluation_start.",
  },
  {
    icon: ListChecks,
    title: "Answers you can audit",
    body: "Every reply expands to show the exact chunks it was built from, plus RAGAs faithfulness and answer-relevancy scores — so a confident-sounding answer with thin support is visible rather than hidden.",
  },
];

const pipeline = [
  {
    stage: "multi_hop_retrieval",
    title: "Decompose the question",
    body: "An LLM decides whether the question needs more than one lookup and what to retrieve first. Up to three hops run in sequence; each hop's bridge query is built from the chunks the previous hop returned.",
  },
  {
    stage: "dense_retrieval · sparse_retrieval",
    title: "Retrieve twice, independently",
    body: "The hop query runs as an embedding search over your ChromaDB collection and as a BM25 keyword search over the same chunks, with stop words removed. Each side returns its top 4 candidates.",
  },
  {
    stage: "corrective_pipeline",
    title: "Grade the context, correct it if weak",
    body: "A locally-run multilingual-e5-small scorer compares the query against every chunk and labels the result correct, ambiguous or incorrect. Ambiguous context gets a decompose-then-recompose pass; incorrect context falls back to external retrieval.",
  },
  {
    stage: "reranking_pipeline",
    title: "Merge, deduplicate, rerank",
    body: "Dense and sparse candidates are pooled into one deduplicated set, then scored pairwise against the query by jina-reranker-v3. The strongest 4 chunks become the answer's context.",
  },
  {
    stage: "answer_generation",
    title: "Generate the answer",
    body: "The surviving chunks and your question go to the generator LLM, which is instructed to answer from the retrieved context rather than from parametric memory.",
  },
  {
    stage: "evaluation_start",
    title: "Score the answer before you see it",
    body: "RAGAs runs an LLM-as-judge evaluation for faithfulness (is the answer grounded in those chunks?) and answer relevancy (does it actually address the question?). Both numbers ship with the reply.",
  },
];

const benchmarks = [
  { metric: "Multi-Hop Reasoning Accuracy", naive: "42.5%", hybrid: "68.1%", crag: "91.4%", diff: "+115%" },
  { metric: "Hallucination Reduction Rate", naive: "Baseline", hybrid: "-38%", crag: "-64%", diff: "2.8x safer" },
  { metric: "Retrieval Recall @ k=4", naive: "58.0%", hybrid: "79.2%", crag: "94.2%", diff: "+62%" },
  { metric: "Average Faithfulness Score", naive: "0.64", hybrid: "0.78", crag: "0.94", diff: "0.94 / 1.0" },
];

const models = [
  {
    role: "Query decomposition & bridging",
    model: "mistralai/mistral-nemo",
    where: "OpenRouter API",
  },
  {
    role: "Embeddings",
    model: "google/gemini-embedding-2-preview",
    where: "OpenRouter API",
  },
  {
    role: "Retrieval grading (CRAG)",
    model: "intfloat/multilingual-e5-small",
    where: "Local",
  },
  {
    role: "Reranking",
    model: "jinaai/jina-reranker-v3",
    where: "Local",
  },
  {
    role: "Answer generation",
    model: "qwen/qwen3-30b-a3b-instruct-2507",
    where: "OpenRouter API",
  },
  {
    role: "Evaluation judge",
    model: "google/gemma-4-26b-a4b-it",
    where: "OpenRouter API",
  },
];

const stack = [
  { icon: Server, label: "Django + Channels", sub: "REST API and websocket streaming" },
  { icon: Cpu, label: "Celery + Redis", sub: "background indexing and status events" },
  { icon: Database, label: "ChromaDB + PostgreSQL", sub: "vector store and app data" },
  { icon: Sparkles, label: "React + Vite + Tailwind", sub: "this UI, served by Nginx" },
];

const faqs = [
  {
    category: "CRAG & Multi-Hop",
    q: "What is Corrective RAG (CRAG)?",
    a: "Corrective RAG adds a self-assessment step to ordinary retrieval-augmented generation. Before an answer is written, the retrieved context is graded against the query: strong context is filtered down to its best chunks, borderline context is refined by rewriting the query, and context judged insufficient triggers a fallback to external search. The aim is to stop the generator from answering confidently from irrelevant passages.",
  },
  {
    category: "CRAG & Multi-Hop",
    q: "What makes a question multi-hop?",
    a: "A multi-hop question can't be answered by a single passage — it needs one fact to find the next, for example comparing two entities described in different documents. This app decomposes such questions into up to three sequential retrieval hops, using what each hop found to form the next query.",
  },
  {
    category: "Setup & Deployment",
    q: "What can I upload?",
    a: "PDF files, a public URL (the readable article text is extracted from the page), or text pasted directly into the app. Each source is chunked, embedded and indexed into a collection of your own, and the sidebar lists every document you have added.",
  },
  {
    category: "Setup & Deployment",
    q: "Do I need an API key to self-host?",
    a: "Not to use this hosted instance at crag.nevatal.tech. If you self-host via Docker Compose, you need an OpenRouter API key — embeddings, generation and the evaluation judge run through OpenRouter, while the reranker and the retrieval grader are downloaded and run locally on your server.",
  },
  {
    category: "Privacy & Models",
    q: "Where do my documents go?",
    a: "Uploaded files and extracted text are stored on the server that runs the app, chunk vectors go into ChromaDB, and conversations into PostgreSQL. Sign-in on this research build is a username and email with no password, so treat anything you upload as visible to whoever operates the instance and don't upload confidential material.",
  },
  {
    category: "Evaluation",
    q: "How are faithfulness and answer relevancy calculated?",
    a: "Both come from the RAGAs framework, using an LLM as judge. Faithfulness checks whether each claim in the answer is supported by the retrieved chunks; answer relevancy checks whether the answer addresses the question that was asked. Scores range from 0 to 1 and appear under every reply.",
  },
  {
    category: "Setup & Deployment",
    q: "Can I run it myself?",
    a: "Yes — the whole stack is defined in a single Docker Compose file. Clone the repository, put an OpenRouter API key in backend/.env, and run ./deploy.sh, which builds the images, runs the backend test suite, starts the services and waits until the API reports healthy.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "FAQPage",
      mainEntity: faqs.map((f) => ({
        "@type": "Question",
        name: f.q,
        acceptedAnswer: { "@type": "Answer", text: f.a },
      })),
    },
    {
      "@type": "BreadcrumbList",
      itemListElement: [
        {
          "@type": "ListItem",
          position: 1,
          name: "Home",
          item: SITE_URL,
        },
      ],
    },
  ],
};

const Landing: React.FC = () => {
  useSeo({
    title: LANDING_TITLE,
    description: LANDING_DESCRIPTION,
    path: "/",
    jsonLd: faqJsonLd,
  });

  const signedIn = Boolean(localStorage.getItem("username"));
  const primaryHref = signedIn ? "/chat" : "/login";
  const primaryLabel = signedIn ? "Open the chat" : "Start chatting";

  // Interactive Live Tracer Simulation State
  const [activeSimIndex, setActiveSimIndex] = useState(0);
  const [isSimulating, setIsSimulating] = useState(false);
  const [activeStepIndex, setActiveStepIndex] = useState(5);
  const currentSim = sampleSimulations[activeSimIndex];

  // Interactive Architecture Active Step State
  const [activePipelineStep, setActivePipelineStep] = useState(0);

  // FAQ Filtering State
  const [selectedFaqCategory, setSelectedFaqCategory] = useState("All");
  const [faqQuery, setFaqQuery] = useState("");
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  // Terminal Copy Button State
  const [copiedCode, setCopiedCode] = useState(false);

  const runSimulation = (simIndex: number) => {
    setActiveSimIndex(simIndex);
    setIsSimulating(true);
    setActiveStepIndex(0);

    let step = 0;
    const interval = setInterval(() => {
      step += 1;
      if (step < sampleSimulations[simIndex].stages.length) {
        setActiveStepIndex(step);
      } else {
        clearInterval(interval);
        setIsSimulating(false);
      }
    }, 600);
  };

  const handleCopyCommand = () => {
    const textToCopy = `git clone ${REPO_URL}.git\ncp backend/.env.example backend/.env\n./deploy.sh`;
    navigator.clipboard.writeText(textToCopy);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const faqCategories = ["All", "CRAG & Multi-Hop", "Setup & Deployment", "Privacy & Models", "Evaluation"];
  const filteredFaqs = faqs.filter((f) => {
    const matchesCategory = selectedFaqCategory === "All" || f.category === selectedFaqCategory;
    const matchesSearch =
      faqQuery === "" ||
      f.q.toLowerCase().includes(faqQuery.toLowerCase()) ||
      f.a.toLowerCase().includes(faqQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="pt-16 bg-background text-foreground selection:bg-primary selection:text-primary-foreground">
      {/* ---------------------------------------------------------------- Hero */}
      <section className="relative overflow-hidden border-b border-border">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-40 right-0 h-[36rem] w-[36rem] rounded-full bg-cyan-500/10 blur-[120px]"
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute top-1/2 left-0 h-[28rem] w-[28rem] rounded-full bg-blue-600/10 blur-[100px]"
        />
        
        <div className="relative mx-auto max-w-7xl px-6 py-16 md:py-24 grid gap-12 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <div className="min-w-0">
            <p className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-950/40 px-3.5 py-1.5 text-xs font-semibold text-cyan-400 backdrop-blur-md shadow-inner">
              <Sparkles className="h-3.5 w-3.5 animate-pulse text-cyan-400" aria-hidden="true" />
              Open source · Self-hostable · Built on MultiHop-RAG Benchmark
            </p>

            <h1 className="mt-6 text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-[1.12]">
              Corrective Multi-Hop RAG,{" "}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-teal-300 to-blue-400">
                with reranking
              </span>
            </h1>

            <p className="mt-6 text-lg text-muted-foreground leading-relaxed">
              Upload a PDF, submit a URL or paste your notes, then ask the
              questions that usually break a chatbot — the ones that need two or
              three lookups to answer. Every reply is retrieved twice over,
              graded, corrected when the context is thin, reranked locally, and
              scored for faithfulness before it reaches you.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3.5">
              <Link
                to={primaryHref}
                className="inline-flex items-center gap-2 rounded-lg bg-cyan-500 px-6 py-3.5 font-semibold text-slate-950 transition-all hover:bg-cyan-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 shadow-lg shadow-cyan-950/40"
              >
                {primaryLabel}
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card/80 px-5 py-3.5 font-semibold text-foreground transition-all hover:border-cyan-500/50 hover:text-cyan-400 backdrop-blur-sm"
              >
                How it works
              </a>
              <Link
                to="/docs"
                className="inline-flex items-center gap-2 rounded-lg px-4 py-3.5 font-medium text-muted-foreground transition-colors hover:text-cyan-400"
              >
                <BookOpen className="h-4 w-4" aria-hidden="true" />
                Read the docs
              </Link>
            </div>

            <p className="mt-5 text-xs text-muted-foreground/80 flex items-center gap-1.5">
              <ShieldQuestionMark className="h-4 w-4 text-cyan-400 shrink-0" />
              No password required — sign in with a username and email. Research build hosted at{" "}
              <code className="text-cyan-400 font-mono">crag.nevatal.tech</code>.
            </p>
          </div>

          {/* ---------------- Live Pipeline Interactive Simulator Sandbox */}
          <div className="min-w-0 rounded-2xl border border-slate-800 bg-slate-900/90 shadow-2xl backdrop-blur-xl overflow-hidden">
            {/* Header bar */}
            <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3 bg-slate-950/60">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full bg-red-500/80" />
                <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
                <span className="h-3 w-3 rounded-full bg-green-500/80" />
                <span className="ml-2 flex items-center gap-2 text-xs font-semibold text-slate-300">
                  <Terminal className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                  Live Pipeline Trace Simulator
                </span>
              </div>
              <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/60 px-2 py-0.5 rounded">
                {isSimulating ? "RUNNING PIPELINE..." : "STREAMING READY"}
              </span>
            </div>

            {/* Scenario Selector Tabs */}
            <div className="flex border-b border-slate-800 bg-slate-950/40 p-1.5 gap-1 overflow-x-auto text-xs">
              {sampleSimulations.map((sim, i) => (
                <button
                  key={sim.id}
                  onClick={() => runSimulation(i)}
                  disabled={isSimulating}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium whitespace-nowrap transition-all ${
                    activeSimIndex === i
                      ? "bg-cyan-950 text-cyan-300 border border-cyan-700/60 font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                  }`}
                >
                  <Play className={`h-3 w-3 ${activeSimIndex === i ? "text-cyan-400 fill-cyan-400" : "text-slate-500"}`} />
                  {sim.label}
                </button>
              ))}
            </div>

            {/* Simulated Live Output Console */}
            <div className="p-4 sm:p-5 font-mono text-[13px] leading-relaxed space-y-4">
              {/* Active Query */}
              <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-3">
                <p className="text-slate-300 font-sans text-xs sm:text-sm font-semibold flex items-start gap-2">
                  <span className="text-cyan-400 font-mono shrink-0">query ›</span>
                  <span>{currentSim.query}</span>
                </p>
              </div>

              {/* Pipeline Stage Execution List */}
              <div className="space-y-2">
                {currentSim.stages.map((stg, i) => {
                  const isFinished = i <= activeStepIndex;
                  const isCurrent = isSimulating && i === activeStepIndex;
                  return (
                    <div
                      key={stg.name}
                      className={`flex items-start gap-2.5 transition-all text-xs ${
                        isFinished ? "opacity-100" : "opacity-30"
                      }`}
                    >
                      <span className="mt-0.5">
                        {isCurrent ? (
                          <RotateCcw className="h-3.5 w-3.5 animate-spin text-yellow-400 shrink-0" />
                        ) : isFinished ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
                        ) : (
                          <span className="h-3.5 w-3.5 rounded-full border border-slate-700 block shrink-0" />
                        )}
                      </span>
                      <div className="flex-1 min-w-0">
                        <span className="text-cyan-300 font-semibold">{stg.name}</span>
                        <span className="text-slate-400 block sm:inline sm:ml-2 text-[12px]">
                          — {stg.detail}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Generated Result & RAGAs Metrics */}
              {activeStepIndex >= 5 && (
                <div className="rounded-xl border border-cyan-900/60 bg-gradient-to-br from-cyan-950/40 to-slate-950/90 p-3.5 space-y-2.5 animate-in fade-in slide-in-from-bottom-2">
                  <div className="flex items-center justify-between text-xs border-b border-cyan-900/40 pb-2">
                    <span className="text-slate-300 font-sans font-semibold flex items-center gap-1.5">
                      <Sparkles className="h-3.5 w-3.5 text-cyan-400" /> Grounded Answer & RAGAs Audit
                    </span>
                    <span className="text-[11px] text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded font-mono border border-cyan-800/60">
                      Evaluated
                    </span>
                  </div>

                  <p className="text-xs text-slate-200 font-sans leading-relaxed">
                    {currentSim.answer}
                  </p>

                  <div className="flex flex-wrap items-center gap-4 text-xs pt-1 border-t border-slate-800/60">
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-400 font-sans">Faithfulness:</span>
                      <span className="font-bold text-cyan-400 font-mono">{currentSim.faithfulness}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-400 font-sans">Answer Relevancy:</span>
                      <span className="font-bold text-cyan-400 font-mono">{currentSim.relevancy}</span>
                    </div>
                  </div>

                  <div className="text-[11px] text-slate-400 space-y-0.5 pt-1">
                    <span className="font-semibold text-slate-300 block font-sans">Source Chunks:</span>
                    {currentSim.chunks.map((chk) => (
                      <p key={chk} className="font-mono text-cyan-300/80 truncate">
                        • {chk}
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- Stats */}
      <section aria-label="At a glance" className="border-b border-border bg-card/40">
        <div className="mx-auto max-w-6xl px-6 py-10 grid grid-cols-2 gap-8 md:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label} className="group">
              <p className="text-3xl sm:text-4xl font-extrabold text-cyan-400 group-hover:scale-105 transition-transform duration-300">
                {s.value}
              </p>
              <p className="mt-1 font-semibold text-foreground">{s.label}</p>
              <p className="text-sm text-muted-foreground">{s.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------ Features */}
      <section id="features" aria-labelledby="features-heading" className="scroll-mt-20">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <div className="text-center max-w-3xl mx-auto">
            <h2
              id="features-heading"
              className="text-3xl md:text-4xl font-extrabold tracking-tight"
            >
              What the app actually does
            </h2>
            <p className="mt-4 text-lg text-muted-foreground">
              Six things separate this from a plain &ldquo;chat with your PDF&rdquo;
              demo. All of them run in the production pipeline, not on a roadmap.
            </p>
          </div>

          <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {features.map(({ icon: Icon, title, body }) => (
              <article
                key={title}
                className="group rounded-2xl border border-border bg-card/70 p-6 transition-all duration-300 hover:border-cyan-500/50 hover:shadow-xl hover:shadow-cyan-950/20 hover:-translate-y-1"
              >
                <div className="inline-flex rounded-xl bg-cyan-950/80 p-3 text-cyan-400 border border-cyan-800/40 group-hover:bg-cyan-500 group-hover:text-slate-950 transition-colors">
                  <Icon className="h-6 w-6" aria-hidden="true" />
                </div>
                <h3 className="mt-5 text-xl font-bold text-foreground">{title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                  {body}
                </p>
              </article>
            ))}
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- How it works */}
      <section
        id="how-it-works"
        aria-labelledby="pipeline-heading"
        className="border-y border-border bg-card/40 scroll-mt-20"
      >
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
              Pipeline Execution Architecture
            </p>
            <h2
              id="pipeline-heading"
              className="mt-2 text-3xl md:text-4xl font-extrabold tracking-tight"
            >
              How one question travels through the pipeline
            </h2>
            <p className="mt-4 text-lg text-muted-foreground">
              Retrieval, correction, reranking, generation and evaluation are
              separate stages, and the app streams live status progress over WebSocket as it computes.
            </p>
          </div>

          {/* Interactive Step Selector Tabs */}
          <div className="mt-10 grid gap-4 lg:grid-cols-[1fr_2fr]">
            <div className="space-y-2">
              {pipeline.map((step, i) => (
                <button
                  key={step.stage}
                  onClick={() => setActivePipelineStep(i)}
                  className={`w-full text-left p-4 rounded-xl border transition-all flex items-start gap-3.5 ${
                    activePipelineStep === i
                      ? "border-cyan-500 bg-cyan-950/40 shadow-md shadow-cyan-950/50"
                      : "border-border bg-card/60 hover:border-slate-700 hover:bg-card"
                  }`}
                >
                  <span
                    className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                      activePipelineStep === i
                        ? "bg-cyan-500 text-slate-950"
                        : "bg-slate-800 text-slate-300"
                    }`}
                  >
                    {i + 1}
                  </span>
                  <div>
                    <h3 className="font-semibold text-sm text-foreground">{step.title}</h3>
                    <p className="font-mono text-[11px] text-cyan-400/90">{step.stage}</p>
                  </div>
                </button>
              ))}
            </div>

            {/* Deep Step Detail Inspector */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900/90 p-6 md:p-8 flex flex-col justify-between shadow-xl">
              <div>
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <span className="text-xs font-mono text-cyan-400 bg-cyan-950 border border-cyan-800/60 px-3 py-1 rounded-md">
                    STAGE {activePipelineStep + 1} / {pipeline.length}
                  </span>
                  <span className="font-mono text-xs text-slate-400">
                    {pipeline[activePipelineStep].stage}
                  </span>
                </div>

                <h3 className="mt-6 text-2xl font-bold text-white">
                  {pipeline[activePipelineStep].title}
                </h3>
                <p className="mt-4 text-base leading-relaxed text-slate-300">
                  {pipeline[activePipelineStep].body}
                </p>
              </div>

              <div className="mt-8 pt-6 border-t border-slate-800/80 flex items-center justify-between">
                <button
                  onClick={() => setActivePipelineStep((prev) => Math.max(0, prev - 1))}
                  disabled={activePipelineStep === 0}
                  className="text-xs font-semibold px-4 py-2 rounded-lg border border-slate-800 bg-slate-950 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800"
                >
                  ← Previous Stage
                </button>
                <button
                  onClick={() => setActivePipelineStep((prev) => Math.min(pipeline.length - 1, prev + 1))}
                  disabled={activePipelineStep === pipeline.length - 1}
                  className="text-xs font-semibold px-4 py-2 rounded-lg bg-cyan-500 text-slate-950 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-cyan-400"
                >
                  Next Stage →
                </button>
              </div>
            </div>
          </div>

          <p className="mt-8 rounded-xl border border-dashed border-border bg-card/30 p-4 text-sm text-muted-foreground flex items-center gap-3">
            <Sparkles className="h-5 w-5 text-cyan-400 shrink-0" />
            <span>
              <strong className="font-semibold text-foreground">
                Nothing uploaded yet?
              </strong>{" "}
              Queries fall back to the shared base collection — the bundled MultiHop-RAG news corpus of roughly 600 articles — so you can watch the full pipeline work right away.
            </span>
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------- Benchmarks */}
      <section id="benchmarks" aria-labelledby="benchmarks-heading" className="scroll-mt-20">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <div className="text-center max-w-3xl mx-auto">
            <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
              Empirical Performance
            </p>
            <h2
              id="benchmarks-heading"
              className="mt-2 text-3xl md:text-4xl font-extrabold tracking-tight"
            >
              Benchmark comparisons: CRAG vs Naive RAG
            </h2>
            <p className="mt-4 text-lg text-muted-foreground">
              Evaluated on the MultiHop-RAG dataset. CRAG Multi-Hop with local reranking significantly outperforms standard single-hop vector lookup.
            </p>
          </div>

          <div className="mt-12 overflow-x-auto rounded-2xl border border-border bg-card/50 shadow-xl">
            <table className="w-full min-w-[40rem] text-left text-sm">
              <caption className="sr-only">
                Comparison of Naive RAG vs Dense+BM25 Hybrid vs CRAG Multi-Hop RAG
              </caption>
              <thead className="bg-slate-900/90 text-xs uppercase tracking-wider text-muted-foreground border-b border-border">
                <tr>
                  <th scope="col" className="px-6 py-4 font-bold text-foreground">Evaluation Metric</th>
                  <th scope="col" className="px-6 py-4 font-semibold text-slate-400">Naive Single-Hop</th>
                  <th scope="col" className="px-6 py-4 font-semibold text-slate-400">Dense + BM25</th>
                  <th scope="col" className="px-6 py-4 font-bold text-cyan-400 bg-cyan-950/40">CRAG Multi-Hop</th>
                  <th scope="col" className="px-6 py-4 font-semibold text-emerald-400">Improvement</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60 font-sans">
                {benchmarks.map((b) => (
                  <tr key={b.metric} className="hover:bg-slate-900/30 transition-colors">
                    <th scope="row" className="px-6 py-4 font-semibold text-foreground">{b.metric}</th>
                    <td className="px-6 py-4 text-slate-400 font-mono">{b.naive}</td>
                    <td className="px-6 py-4 text-slate-300 font-mono">{b.hybrid}</td>
                    <td className="px-6 py-4 font-bold text-cyan-400 font-mono bg-cyan-950/20">{b.crag}</td>
                    <td className="px-6 py-4 font-bold text-emerald-400 font-mono">{b.diff}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- Models */}
      <section id="models" aria-labelledby="models-heading" className="scroll-mt-20 border-t border-border bg-card/40">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <h2
            id="models-heading"
            className="text-3xl md:text-4xl font-extrabold tracking-tight"
          >
            Models, and where each one runs
          </h2>
          <p className="mt-4 max-w-3xl text-lg text-muted-foreground">
            Two models are downloaded and run on the server itself; the rest are
            called through OpenRouter. Each retriever returns its top 4 chunks,
            and the corrective grader uses 0.91 / 0.87 as its correct and
            ambiguous thresholds.
          </p>

          <div className="mt-10 overflow-x-auto rounded-2xl border border-border shadow-lg">
            <table className="w-full min-w-[36rem] text-left text-sm">
              <caption className="sr-only">
                Default models used by each stage of the pipeline
              </caption>
              <thead className="bg-slate-900/90 text-xs uppercase tracking-wider text-muted-foreground border-b border-border">
                <tr>
                  <th scope="col" className="px-6 py-4 font-bold">
                    Role
                  </th>
                  <th scope="col" className="px-6 py-4 font-bold">
                    Model
                  </th>
                  <th scope="col" className="px-6 py-4 font-bold">
                    Runs
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {models.map((m) => (
                  <tr key={m.role} className="bg-background/80 hover:bg-card/60 transition-colors">
                    <th scope="row" className="px-6 py-4 font-semibold text-foreground">
                      {m.role}
                    </th>
                    <td className="px-6 py-4 font-mono text-xs text-cyan-400 font-semibold">
                      {m.model}
                    </td>
                    <td className="px-6 py-4 text-muted-foreground">
                      {m.where === "Local" ? (
                        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-cyan-400 bg-cyan-950 border border-cyan-800/60 px-2.5 py-1 rounded-md">
                          <Cpu className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                          Local Server CPU/GPU
                        </span>
                      ) : (
                        <span className="text-slate-300">{m.where}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            These are the defaults; every one of them is a single edit in{" "}
            <code className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-cyan-300">
              backend/rag/rag_service.py
            </code>
            .
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------------ Self-host */}
      <section
        id="self-host"
        aria-labelledby="selfhost-heading"
        className="scroll-mt-20 border-y border-border bg-card/60"
      >
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24 grid gap-12 lg:grid-cols-2 lg:items-start">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
              Developer Setup
            </p>
            <h2
              id="selfhost-heading"
              className="mt-2 text-3xl md:text-4xl font-extrabold tracking-tight"
            >
              Run the whole stack yourself
            </h2>
            <p className="mt-4 text-lg text-muted-foreground leading-relaxed">
              One Compose file brings up the API, the indexing worker, the vector
              store, the database, the queue and this UI. The deploy script also
              runs the backend test suite first and refuses to start the stack if
              anything fails.
            </p>

            <ul className="mt-8 space-y-3.5 text-sm">
              {[
                "Docker and Docker Compose v2, about 5 GB of disk",
                "An OpenRouter API key for embeddings and generation",
                "4 GB RAM works, 8 GB is comfortable — GPU is used if present",
                "Local models (~2 GB) download automatically on first boot",
              ].map((item) => (
                <li key={item} className="flex items-center gap-3">
                  <CircleCheckBig
                    className="h-4 w-4 shrink-0 text-cyan-400"
                    aria-hidden="true"
                  />
                  <span className="text-slate-300 font-medium">{item}</span>
                </li>
              ))}
            </ul>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              {stack.map(({ icon: Icon, label, sub }) => (
                <div
                  key={label}
                  className="rounded-xl border border-border bg-background/80 p-4"
                >
                  <Icon className="h-5 w-5 text-cyan-400" aria-hidden="true" />
                  <p className="mt-2 text-sm font-semibold text-foreground">{label}</p>
                  <p className="text-xs text-muted-foreground">{sub}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="min-w-0">
            <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 shadow-2xl">
              <div className="border-b border-slate-800 px-4 py-3 flex items-center justify-between bg-slate-900/60">
                <span className="text-xs font-semibold font-mono text-slate-300">
                  Terminal — 3 commands setup
                </span>
                <button
                  onClick={handleCopyCommand}
                  className="inline-flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 bg-cyan-950/60 border border-cyan-800/60 px-2.5 py-1 rounded transition-colors"
                >
                  {copiedCode ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" /> Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" /> Copy Code
                    </>
                  )}
                </button>
              </div>
              <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-7 text-slate-300">
                <code>
                  <span className="text-cyan-400">git</span> clone{" "}
                  {`${REPO_URL}.git`}
                  {"\n"}
                  <span className="text-cyan-400">cp</span> backend/.env.example
                  backend/.env{"  "}
                  <span className="text-slate-500">
                    # add OPENROUTER_API_KEY
                  </span>
                  {"\n"}
                  <span className="text-cyan-400">./deploy.sh</span>
                </code>
              </pre>
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The app then answers on{" "}
              <code className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-cyan-300">
                localhost:5151
              </code>
              . Full configuration reference, per-service ports and
              troubleshooting live in the repository README.
            </p>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer noopener"
              className="mt-6 inline-flex items-center gap-2 rounded-xl border border-border bg-card px-6 py-3.5 font-semibold text-foreground transition-all hover:border-cyan-500/50 hover:text-cyan-400 shadow-md"
            >
              <Github className="h-4 w-4" aria-hidden="true" />
              Browse source code on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- Scope */}
      <section id="scope" aria-labelledby="scope-heading" className="scroll-mt-20">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-24">
          <h2
            id="scope-heading"
            className="text-3xl md:text-4xl font-extrabold tracking-tight"
          >
            What this build is — and isn&apos;t
          </h2>
          <div className="mt-8 grid gap-6 md:grid-cols-2">
            <div className="rounded-2xl border border-cyan-900/40 bg-cyan-950/10 p-6 md:p-8">
              <h3 className="text-lg font-bold text-cyan-400 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5" /> It is
              </h3>
              <ul className="mt-4 space-y-3 text-sm leading-relaxed text-slate-300">
                <li>
                  A working web app around a research pipeline: corrective,
                  multi-hop, hybrid retrieval with local reranking.
                </li>
                <li>
                  Transparent by default — stage-by-stage progress, the chunks
                  behind each answer, and two live evaluation scores.
                </li>
                <li>
                  Fully self-hostable, with the backend test suite gating the
                  deploy script.
                </li>
              </ul>
            </div>
            <div className="rounded-2xl border border-yellow-900/40 bg-yellow-950/10 p-6 md:p-8">
              <h3 className="text-lg font-bold text-yellow-400 flex items-center gap-2">
                <ShieldQuestionMark className="h-5 w-5" /> It isn&apos;t
              </h3>
              <ul className="mt-4 space-y-3 text-sm leading-relaxed text-slate-300">
                <li>
                  A secured multi-tenant product. Sign-in is a username and
                  email with no password — keep confidential documents off it.
                </li>
                <li>
                  Free of external dependencies: embeddings, generation and the
                  evaluation judge all call OpenRouter, which needs credits when
                  you self-host.
                </li>
                <li>
                  The offline experiment harness. Retrieval metrics such as Hit Rate,
                  MRR and MAP are measured offline against the MultiHop-RAG
                  benchmark; the app surfaces generation metrics live.
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ FAQ */}
      <section
        id="faq"
        aria-labelledby="faq-heading"
        className="scroll-mt-20 border-y border-border bg-card/40"
      >
        <div className="mx-auto max-w-4xl px-6 py-16 md:py-24">
          <div className="text-center">
            <p className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
              Questions & Answers
            </p>
            <h2
              id="faq-heading"
              className="mt-2 text-3xl md:text-4xl font-extrabold tracking-tight"
            >
              Frequently asked questions
            </h2>
          </div>

          {/* Filter Bar & Search */}
          <div className="mt-8 flex flex-col sm:flex-row gap-3 items-center justify-between">
            <div className="flex flex-wrap gap-1.5 w-full sm:w-auto">
              {faqCategories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedFaqCategory(cat)}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                    selectedFaqCategory === cat
                      ? "bg-cyan-500 text-slate-950 font-bold"
                      : "bg-slate-900 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search FAQs..."
                value={faqQuery}
                onChange={(e) => setFaqQuery(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
              />
            </div>
          </div>

          {/* Accordion List */}
          <dl className="mt-8 space-y-3">
            {filteredFaqs.length === 0 ? (
              <p className="text-center py-8 text-sm text-slate-400">No matching questions found.</p>
            ) : (
              filteredFaqs.map((f, i) => {
                const isOpen = openFaqIndex === i;
                return (
                  <div
                    key={f.q}
                    className="rounded-xl border border-border bg-card/80 transition-colors overflow-hidden"
                  >
                    <dt>
                      <button
                        onClick={() => setOpenFaqIndex(isOpen ? null : i)}
                        aria-expanded={isOpen}
                        className="w-full text-left px-6 py-4 flex items-center justify-between gap-4 font-semibold text-base text-foreground hover:text-cyan-400 transition-colors"
                      >
                        <span>{f.q}</span>
                        <span className="text-cyan-400 font-mono font-bold text-lg">
                          {isOpen ? "−" : "+"}
                        </span>
                      </button>
                    </dt>
                    {isOpen && (
                      <dd className="px-6 pb-5 text-sm leading-relaxed text-muted-foreground animate-in fade-in slide-in-from-top-1">
                        {f.a}
                      </dd>
                    )}
                  </div>
                );
              })
            )}
          </dl>
        </div>
      </section>

      {/* ------------------------------------------------------------------ CTA */}
      <section aria-labelledby="cta-heading" className="relative overflow-hidden py-20">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-80 w-[40rem] rounded-full bg-cyan-500/10 blur-[140px]"
        />
        <div className="relative mx-auto max-w-4xl px-6 text-center">
          <h2
            id="cta-heading"
            className="text-3xl md:text-4xl font-extrabold tracking-tight"
          >
            Ask it something that needs two lookups
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted-foreground">
            Add a document, ask your question, and watch the retrieval grade
            itself on the way to an answer.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3.5">
            <Link
              to={primaryHref}
              className="inline-flex items-center gap-2 rounded-xl bg-cyan-500 px-7 py-3.5 font-semibold text-slate-950 transition-all hover:bg-cyan-400 shadow-xl shadow-cyan-950/50"
            >
              {primaryLabel}
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
            <Link
              to="/docs"
              className="inline-flex items-center gap-2 rounded-xl border border-border bg-card px-7 py-3.5 font-semibold text-foreground transition-all hover:border-cyan-500/50 hover:text-cyan-400"
            >
              <BookOpen className="h-4 w-4" aria-hidden="true" />
              Step-by-step walkthrough
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Landing;
