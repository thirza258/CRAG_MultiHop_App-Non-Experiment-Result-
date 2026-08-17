import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Github } from "lucide-react";
import { useSeo, REPO_URL } from "../lib/seo";
import Footer from "../components/Footer";

const layers = [
  {
    n: "01",
    title: "Core retriever",
    body: "Dense retrieval over embeddings, sparse BM25 retrieval over the same chunks, and a hybrid stage that merges both candidate sets, removes duplicates and reranks what remains with a cross-encoder.",
  },
  {
    n: "02",
    title: "Corrective RAG (CRAG)",
    body: "Retrieved context is self-graded as correct, ambiguous or incorrect. Correct context is filtered down to its strongest chunks, ambiguous context is refined by decomposing and recomposing the query, and context judged insufficient escalates to fallback retrieval outside the index.",
  },
  {
    n: "03",
    title: "Multi-hop orchestrator",
    body: "Questions whose answer is spread across several documents are decomposed into sequential retrieval hops — up to three — where each hop's bridge query is formed from what earlier hops retrieved.",
  },
];

const metrics = [
  {
    group: "Retrieval (measured offline)",
    items: ["Hit Rate", "Recall@k", "Precision@k", "MRR", "MAP"],
  },
  {
    group: "Generation",
    items: [
      "BERTScore",
      "RAGAs Faithfulness",
      "RAGAs Answer Relevancy",
      "RAGAs Answer Correctness",
    ],
  },
];

const references = [
  {
    text: "Yan et al. — Corrective Retrieval-Augmented Generation",
    note: "the self-grading and correction layer",
  },
  {
    text: "Tang et al. — MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries",
    note: "the benchmark and the bundled news corpus",
  },
  {
    text: "RAGAs evaluation framework, and BERTScore",
    note: "the generation metrics",
  },
];

const About: React.FC = () => {
  useSeo({
    title: "About the project | CRAG MultiHop RAG",
    description:
      "The research behind this app: a three-layer RAG pipeline combining hybrid retrieval with reranking, Corrective RAG self-grading, and multi-hop query decomposition, evaluated on the MultiHop-RAG benchmark.",
    path: "/about",
  });

  return (
    <div className="pt-16 bg-background text-foreground">
      <div className="mx-auto max-w-4xl px-6 py-16 md:py-20">
        <p className="text-sm font-semibold uppercase tracking-wider text-primary">
          About
        </p>
        <h1 className="mt-4 text-4xl md:text-5xl font-bold tracking-tight leading-tight">
          A research pipeline you can talk to
        </h1>
        <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
          This project implements and evaluates an integrated
          retrieval-augmented generation pipeline that stacks three layers on
          top of each other, then wraps the result in a web app so the behaviour
          of each layer is visible while a question is being answered. It was
          built as a research project on the{" "}
          <strong className="font-semibold text-foreground">MultiHop-RAG</strong>{" "}
          benchmark, and the deployed instance runs the same pipeline as the
          experiments.
        </p>

        <section aria-labelledby="layers-heading" className="mt-16">
          <h2 id="layers-heading" className="text-2xl font-bold tracking-tight">
            The three layers
          </h2>
          <div className="mt-8 space-y-4">
            {layers.map((l) => (
              <article
                key={l.n}
                className="rounded-xl border border-border bg-card p-6"
              >
                <div className="flex items-baseline gap-4">
                  <span className="font-mono text-sm font-bold text-primary">
                    {l.n}
                  </span>
                  <h3 className="text-lg font-semibold">{l.title}</h3>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                  {l.body}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section aria-labelledby="metrics-heading" className="mt-16">
          <h2 id="metrics-heading" className="text-2xl font-bold tracking-tight">
            How it is evaluated
          </h2>
          <p className="mt-4 leading-relaxed text-muted-foreground">
            The pipeline is scored on the MultiHop-RAG benchmark across both
            retrieval and generation. Of these, the web app computes and displays
            two per answer — faithfulness and answer relevancy — because they can
            be measured without a ground-truth reference.
          </p>
          <div className="mt-8 grid gap-6 sm:grid-cols-2">
            {metrics.map((m) => (
              <div
                key={m.group}
                className="rounded-xl border border-border bg-card p-6"
              >
                <h3 className="text-sm font-semibold uppercase tracking-wider text-primary">
                  {m.group}
                </h3>
                <ul className="mt-4 space-y-2 text-sm text-muted-foreground">
                  {m.items.map((item) => (
                    <li key={item} className="font-mono text-xs">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        <section aria-labelledby="honesty-heading" className="mt-16">
          <h2 id="honesty-heading" className="text-2xl font-bold tracking-tight">
            Scope of this deployment
          </h2>
          <ul className="mt-6 space-y-4 leading-relaxed text-muted-foreground">
            <li>
              Sign-in takes a username and an email address and sets no password,
              so per-user document collections separate your data from other
              people&apos;s but do not protect it. Please don&apos;t upload
              confidential material.
            </li>
            <li>
              Embeddings, answer generation and the evaluation judge are called
              through OpenRouter; the reranker and the retrieval grader are
              downloaded once and run on the server. Answers therefore depend on
              third-party model availability.
            </li>
            <li>
              The offline experiment harness — the retrieval metrics, the
              benchmark runs, the ground-truth comparisons — is not part of this
              web app. What you see here is the pipeline itself, plus live
              generation scoring.
            </li>
          </ul>
        </section>

        <section aria-labelledby="refs-heading" className="mt-16">
          <h2 id="refs-heading" className="text-2xl font-bold tracking-tight">
            References
          </h2>
          <ul className="mt-6 space-y-3">
            {references.map((r) => (
              <li key={r.text} className="text-sm text-muted-foreground">
                <span className="text-foreground">{r.text}</span>
                <span className="block text-xs text-muted-foreground/80">
                  {r.note}
                </span>
              </li>
            ))}
          </ul>
        </section>

        <div className="mt-16 flex flex-wrap gap-3">
          <Link
            to="/docs"
            className="inline-flex items-center gap-2 rounded-md bg-primary px-5 py-3 font-semibold text-primary-foreground transition-colors hover:bg-primary-hover"
          >
            See the walkthrough
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex items-center gap-2 rounded-md border border-border bg-card px-5 py-3 font-semibold transition-colors hover:border-primary/50 hover:text-primary"
          >
            <Github className="h-4 w-4" aria-hidden="true" />
            Read the source
          </a>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default About;
