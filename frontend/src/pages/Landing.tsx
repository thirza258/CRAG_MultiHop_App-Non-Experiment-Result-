import React from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BookOpen,
  CircleCheckBig,
  Cpu,
  Database,
  FileText,
  GitBranch,
  Github,
  Layers,
  ListChecks,
  Server,
  ShieldQuestionMark,
  Sparkles,
  Terminal,
} from "lucide-react";
import { useSeo, REPO_URL } from "../lib/seo";
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

const features = [
  {
    icon: FileText,
    title: "Bring your own documents",
    body: "Upload a PDF, submit a URL, or paste raw text. The text is extracted, split into ~500-character overlapping chunks, embedded, and stored in ChromaDB by a background worker, so a long document never blocks the UI.",
  },
  {
    icon: GitBranch,
    title: "Questions that need more than one lookup",
    body: "The multi-hop orchestrator decides whether your question needs follow-up retrieval, then runs up to three sequential hops — each one building its query from what the previous hop found, and stopping early once a hop adds nothing new.",
  },
  {
    icon: ShieldQuestionMark,
    title: "Retrieval that grades itself",
    body: "A Corrective RAG layer scores the retrieved context as correct, ambiguous or incorrect. Correct context is stripped of weak chunks, ambiguous context is refined with an expanded query, and context judged insufficient escalates to an external search.",
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
    q: "What is Corrective RAG (CRAG)?",
    a: "Corrective RAG adds a self-assessment step to ordinary retrieval-augmented generation. Before an answer is written, the retrieved context is graded against the query: strong context is filtered down to its best chunks, borderline context is refined by rewriting the query, and context judged insufficient triggers a fallback to external search. The aim is to stop the generator from answering confidently from irrelevant passages.",
  },
  {
    q: "What makes a question multi-hop?",
    a: "A multi-hop question can't be answered by a single passage — it needs one fact to find the next, for example comparing two entities described in different documents. This app decomposes such questions into up to three sequential retrieval hops, using what each hop found to form the next query.",
  },
  {
    q: "What can I upload?",
    a: "PDF files, a public URL (the readable article text is extracted from the page), or text pasted directly into the app. Each source is chunked, embedded and indexed into a collection of your own, and the sidebar lists every document you have added.",
  },
  {
    q: "Do I need an API key?",
    a: "Not to use this hosted instance. If you self-host, you need an OpenRouter API key — embeddings, generation and the evaluation judge run through OpenRouter, while the reranker and the retrieval grader are downloaded and run locally.",
  },
  {
    q: "Where do my documents go?",
    a: "Uploaded files and extracted text are stored on the server that runs the app, chunk vectors go into ChromaDB, and conversations into PostgreSQL. Sign-in on this research build is a username and email with no password, so treat anything you upload as visible to whoever operates the instance and don't upload confidential material.",
  },
  {
    q: "How are faithfulness and answer relevancy calculated?",
    a: "Both come from the RAGAs framework, using an LLM as judge. Faithfulness checks whether each claim in the answer is supported by the retrieved chunks; answer relevancy checks whether the answer addresses the question that was asked. Scores range from 0 to 1 and appear under every reply.",
  },
  {
    q: "Can I run it myself?",
    a: "Yes — the whole stack is defined in a single Docker Compose file. Clone the repository, put an OpenRouter API key in backend/.env, and run ./deploy.sh, which builds the images, runs the backend test suite, starts the services and waits until the API reports healthy.",
  },
];

const faqJsonLd = {
  "@context": "https://schema.org",
  "@type": "FAQPage",
  mainEntity: faqs.map((f) => ({
    "@type": "Question",
    name: f.q,
    acceptedAnswer: { "@type": "Answer", text: f.a },
  })),
};

const Landing: React.FC = () => {
  useSeo({
    title: LANDING_TITLE,
    description: LANDING_DESCRIPTION,
    path: "/",
  });

  const signedIn = Boolean(localStorage.getItem("username"));
  const primaryHref = signedIn ? "/chat" : "/login";
  const primaryLabel = signedIn ? "Open the chat" : "Start chatting";

  return (
    <div className="pt-16 bg-background text-foreground">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
      />

      {/* ---------------------------------------------------------------- Hero */}
      <section className="relative overflow-hidden border-b border-border">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-40 right-0 h-[32rem] w-[32rem] rounded-full bg-primary/10 blur-3xl"
        />
        <div className="relative mx-auto max-w-6xl px-6 py-16 md:py-24 grid gap-12 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
          <div className="min-w-0">
            <p className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-primary">
              <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
              Open source · self-hostable · built on the MultiHop-RAG benchmark
            </p>

            <h1 className="mt-6 text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight leading-[1.1]">
              Corrective Multi-Hop RAG,{" "}
              <span className="text-primary">with reranking</span>
            </h1>

            <p className="mt-6 text-lg text-muted-foreground leading-relaxed">
              Upload a PDF, submit a URL or paste your notes, then ask the
              questions that usually break a chatbot — the ones that need two or
              three lookups to answer. Every reply is retrieved twice over,
              graded, corrected when the context is thin, reranked locally, and
              scored for faithfulness before it reaches you.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                to={primaryHref}
                className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-3 font-semibold text-primary-foreground transition-colors hover:bg-primary-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              >
                {primaryLabel}
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </Link>
              <a
                href="#how-it-works"
                className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-5 py-3 font-semibold text-foreground transition-colors hover:border-primary/50 hover:text-primary"
              >
                How it works
              </a>
              <Link
                to="/docs"
                className="inline-flex items-center gap-2 rounded-md px-3 py-3 font-medium text-muted-foreground transition-colors hover:text-primary"
              >
                <BookOpen className="h-4 w-4" aria-hidden="true" />
                Read the docs
              </Link>
            </div>

            <p className="mt-5 text-sm text-muted-foreground">
              No password to remember — sign in with a username and email. It is
              a research build, so please don&apos;t upload anything
              confidential.
            </p>
          </div>

          {/* Pipeline trace mock-up: mirrors the websocket stages the app streams */}
          <div className="min-w-0 rounded-xl border border-border bg-card shadow-2xl">
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <span className="h-3 w-3 rounded-full bg-red-500/80" />
              <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
              <span className="h-3 w-3 rounded-full bg-green-500/80" />
              <span className="ml-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Terminal className="h-3.5 w-3.5" aria-hidden="true" />
                live pipeline trace
              </span>
            </div>
            <div className="space-y-3 p-4 font-mono text-[13px] leading-relaxed">
              <p className="text-foreground">
                <span className="text-primary">query&nbsp;›</span> Which of the
                two reports flagged the same supplier risk, and what did each
                recommend?
              </p>
              <ul className="space-y-2 text-muted-foreground">
                <li>
                  <span className="text-blue-400">multi_hop_retrieval</span> —
                  starting hop 1
                </li>
                <li>
                  <span className="text-blue-400">dense_retrieval</span> — 4
                  chunks · <span className="text-blue-400">sparse_retrieval</span>{" "}
                  — 4 chunks
                </li>
                <li>
                  <span className="text-blue-400">corrective_pipeline</span> —
                  decision: ambiguous, local_score 0.883 → refining
                </li>
                <li>
                  <span className="text-blue-400">multi_hop_retrieval</span> —
                  bridge query for hop 2
                </li>
                <li>
                  <span className="text-blue-400">reranking_pipeline</span> —
                  merged 8 candidates into 6 unique, kept top 4
                </li>
                <li>
                  <span className="text-blue-400">answer_generation</span> —
                  answer generated by LLM
                </li>
              </ul>
              <div className="rounded-lg border border-border bg-background p-3">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">
                  evaluation
                </p>
                <p className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-foreground">
                  <span>
                    faithfulness{" "}
                    <span className="font-semibold text-primary">0.94</span>
                  </span>
                  <span>
                    answer relevancy{" "}
                    <span className="font-semibold text-primary">0.91</span>
                  </span>
                </p>
              </div>
              <p className="text-xs text-muted-foreground">
                Illustrative trace. Stage names are the ones the app emits.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------------- Stats */}
      <section aria-label="At a glance" className="border-b border-border bg-card/40">
        <div className="mx-auto max-w-6xl px-6 py-10 grid grid-cols-2 gap-8 md:grid-cols-4">
          {stats.map((s) => (
            <div key={s.label}>
              <p className="text-3xl font-bold text-primary">{s.value}</p>
              <p className="mt-1 font-medium text-foreground">{s.label}</p>
              <p className="text-sm text-muted-foreground">{s.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------ Features */}
      <section id="features" aria-labelledby="features-heading" className="scroll-mt-20">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-20">
          <h2
            id="features-heading"
            className="text-3xl md:text-4xl font-bold tracking-tight"
          >
            What the app actually does
          </h2>
          <p className="mt-4 max-w-3xl text-lg text-muted-foreground">
            Six things separate this from a plain &ldquo;chat with your PDF&rdquo;
            demo. All of them are in the running pipeline, not on a roadmap.
          </p>

          <div className="mt-10 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {features.map(({ icon: Icon, title, body }) => (
              <article
                key={title}
                className="rounded-xl border border-border bg-card p-6 transition-colors hover:border-primary/50"
              >
                <div className="inline-flex rounded-lg bg-accent p-2.5 text-primary">
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </div>
                <h3 className="mt-4 text-lg font-semibold">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
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
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-20">
          <h2
            id="pipeline-heading"
            className="text-3xl md:text-4xl font-bold tracking-tight"
          >
            How one question travels through the pipeline
          </h2>
          <p className="mt-4 max-w-3xl text-lg text-muted-foreground">
            Retrieval, correction, reranking, generation and evaluation are
            separate stages, and the app tells you which one it is in while you
            wait.
          </p>

          <ol className="mt-10 space-y-4">
            {pipeline.map((step, i) => (
              <li
                key={step.stage}
                className="grid gap-4 rounded-xl border border-border bg-background p-6 md:grid-cols-[auto_1fr] md:gap-6"
              >
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-border bg-card font-bold text-primary">
                  {i + 1}
                </span>
                <div>
                  <h3 className="text-lg font-semibold">{step.title}</h3>
                  <p className="mt-1 font-mono text-xs text-blue-400">
                    {step.stage}
                  </p>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                    {step.body}
                  </p>
                </div>
              </li>
            ))}
          </ol>

          <p className="mt-8 rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground">
            <strong className="font-semibold text-foreground">
              Nothing uploaded yet?
            </strong>{" "}
            Queries fall back to the instance&apos;s shared base collection — the
            bundled MultiHop-RAG news corpus of roughly 600 articles, when the
            operator has indexed it — so you can watch the pipeline work before
            adding a document of your own.
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------------- Models */}
      <section id="models" aria-labelledby="models-heading" className="scroll-mt-20">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-20">
          <h2
            id="models-heading"
            className="text-3xl md:text-4xl font-bold tracking-tight"
          >
            Models, and where each one runs
          </h2>
          <p className="mt-4 max-w-3xl text-lg text-muted-foreground">
            Two models are downloaded and run on the server itself; the rest are
            called through OpenRouter. Each retriever returns its top 4 chunks,
            and the corrective grader uses 0.91 / 0.87 as its correct and
            ambiguous thresholds.
          </p>

          <div className="mt-10 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[36rem] text-left text-sm">
              <caption className="sr-only">
                Default models used by each stage of the pipeline
              </caption>
              <thead className="bg-card text-xs uppercase tracking-wider text-muted-foreground">
                <tr>
                  <th scope="col" className="px-5 py-3 font-semibold">
                    Role
                  </th>
                  <th scope="col" className="px-5 py-3 font-semibold">
                    Model
                  </th>
                  <th scope="col" className="px-5 py-3 font-semibold">
                    Runs
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {models.map((m) => (
                  <tr key={m.role} className="bg-background/60">
                    <th scope="row" className="px-5 py-3 font-medium">
                      {m.role}
                    </th>
                    <td className="px-5 py-3 font-mono text-xs text-primary">
                      {m.model}
                    </td>
                    <td className="px-5 py-3 text-muted-foreground">
                      {m.where === "Local" ? (
                        <span className="inline-flex items-center gap-1.5 text-foreground">
                          <Cpu className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
                          Local
                        </span>
                      ) : (
                        m.where
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            These are the defaults; every one of them is a single edit in{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground">
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
        className="scroll-mt-20 border-y border-border bg-card/40"
      >
        {/* min-w-0 on both columns: the wide <pre> below would otherwise force
            the grid track past the viewport on narrow screens. */}
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-20 grid gap-12 lg:grid-cols-2 lg:items-start">
          <div className="min-w-0">
            <h2
              id="selfhost-heading"
              className="text-3xl md:text-4xl font-bold tracking-tight"
            >
              Run the whole stack yourself
            </h2>
            <p className="mt-4 text-lg text-muted-foreground">
              One Compose file brings up the API, the indexing worker, the vector
              store, the database, the queue and this UI. The deploy script also
              runs the backend test suite first and refuses to start the stack if
              anything fails.
            </p>

            <ul className="mt-8 space-y-3 text-sm">
              {[
                "Docker and Docker Compose v2, about 5 GB of disk",
                "An OpenRouter API key for embeddings and generation",
                "4 GB RAM works, 8 GB is comfortable — a GPU is used if present, never required",
                "Local models (~2 GB) download on first boot, resumably",
              ].map((item) => (
                <li key={item} className="flex gap-3">
                  <CircleCheckBig
                    className="mt-0.5 h-4 w-4 shrink-0 text-primary"
                    aria-hidden="true"
                  />
                  <span className="text-muted-foreground">{item}</span>
                </li>
              ))}
            </ul>

            <div className="mt-8 grid gap-4 sm:grid-cols-2">
              {stack.map(({ icon: Icon, label, sub }) => (
                <div
                  key={label}
                  className="rounded-lg border border-border bg-background p-4"
                >
                  <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
                  <p className="mt-2 text-sm font-semibold">{label}</p>
                  <p className="text-xs text-muted-foreground">{sub}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="min-w-0">
            <div className="overflow-hidden rounded-xl border border-border bg-background">
              <div className="border-b border-border px-4 py-2.5 text-xs font-medium text-muted-foreground">
                three commands
              </div>
              <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-7 text-muted-foreground">
                <code>
                  <span className="text-primary">git</span> clone{" "}
                  {`${REPO_URL}.git`}
                  {"\n"}
                  <span className="text-primary">cp</span> backend/.env.example
                  backend/.env{"  "}
                  <span className="text-muted-foreground/70">
                    # add OPENROUTER_API_KEY
                  </span>
                  {"\n"}
                  <span className="text-primary">./deploy.sh</span>
                </code>
              </pre>
            </div>
            <p className="mt-4 text-sm text-muted-foreground">
              The app then answers on{" "}
              <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground">
                localhost:5151
              </code>
              . Full configuration reference, per-service ports and
              troubleshooting live in the repository README.
            </p>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer noopener"
              className="mt-6 inline-flex items-center gap-2 rounded-md border border-border bg-card px-5 py-3 font-semibold transition-colors hover:border-primary/50 hover:text-primary"
            >
              <Github className="h-4 w-4" aria-hidden="true" />
              Browse the source
            </a>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- Limitations */}
      <section id="scope" aria-labelledby="scope-heading" className="scroll-mt-20">
        <div className="mx-auto max-w-6xl px-6 py-16 md:py-20">
          <h2
            id="scope-heading"
            className="text-3xl md:text-4xl font-bold tracking-tight"
          >
            What this build is — and isn&apos;t
          </h2>
          <div className="mt-8 grid gap-6 md:grid-cols-2">
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="font-semibold text-primary">It is</h3>
              <ul className="mt-4 space-y-3 text-sm leading-relaxed text-muted-foreground">
                <li>
                  A working web app around a research pipeline: corrective,
                  multi-hop, hybrid retrieval with local reranking.
                </li>
                <li>
                  Transparent by default — stage-by-stage progress, the chunks
                  behind each answer, and two evaluation scores.
                </li>
                <li>
                  Fully self-hostable, with the backend test suite gating the
                  deploy.
                </li>
              </ul>
            </div>
            <div className="rounded-xl border border-border bg-card p-6">
              <h3 className="font-semibold text-yellow-500">It isn&apos;t</h3>
              <ul className="mt-4 space-y-3 text-sm leading-relaxed text-muted-foreground">
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
                  The full research harness. Retrieval metrics such as Hit Rate,
                  MRR and MAP are measured offline against the MultiHop-RAG
                  benchmark; the app surfaces the two generation metrics live.
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
        <div className="mx-auto max-w-4xl px-6 py-16 md:py-20">
          <h2
            id="faq-heading"
            className="text-3xl md:text-4xl font-bold tracking-tight"
          >
            Frequently asked questions
          </h2>
          <dl className="mt-10 space-y-4">
            {faqs.map((f) => (
              <div
                key={f.q}
                className="rounded-xl border border-border bg-background p-6"
              >
                <dt className="text-lg font-semibold">{f.q}</dt>
                <dd className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {f.a}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* ------------------------------------------------------------------ CTA */}
      <section aria-labelledby="cta-heading">
        <div className="mx-auto max-w-4xl px-6 py-20 text-center">
          <h2
            id="cta-heading"
            className="text-3xl md:text-4xl font-bold tracking-tight"
          >
            Ask it something that needs two lookups
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted-foreground">
            Add a document, ask your question, and watch the retrieval grade
            itself on the way to an answer.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              to={primaryHref}
              className="inline-flex items-center gap-2 rounded-md bg-primary px-6 py-3 font-semibold text-primary-foreground transition-colors hover:bg-primary-hover"
            >
              {primaryLabel}
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
            <Link
              to="/docs"
              className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-6 py-3 font-semibold transition-colors hover:border-primary/50 hover:text-primary"
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
