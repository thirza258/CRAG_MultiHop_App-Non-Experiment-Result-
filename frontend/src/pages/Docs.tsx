import React, { useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { Link2 } from "lucide-react";
import { steps } from "../components/data/DocsData";
import { useSeo, SITE_URL } from "../lib/seo";
import Footer from "../components/Footer";

const Docs: React.FC = () => {
  const { slug } = useParams<{ slug?: string }>();
  const activeStep = slug ? steps.find((s) => s.slug === slug) : undefined;

  const title = activeStep
    ? `${activeStep.title} — Step ${activeStep.id} | CRAG MultiHop RAG Docs`
    : "How to use the app, step by step | CRAG MultiHop RAG";

  const description = activeStep
    ? `Step ${activeStep.id}: ${activeStep.title}. Walkthrough and usage guide for the CRAG MultiHop RAG pipeline.`
    : "A walkthrough of the app: signing in, adding a PDF, URL or pasted text, waiting for indexing, chatting with the corrective multi-hop pipeline, and reading the retrieved chunks and evaluation scores.";

  const path = activeStep ? `/docs/${activeStep.slug}` : "/docs";

  const breadcrumbsJsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      {
        "@type": "ListItem",
        position: 1,
        name: "Home",
        item: SITE_URL,
      },
      {
        "@type": "ListItem",
        position: 2,
        name: "Documentation",
        item: `${SITE_URL}/docs`,
      },
      ...(activeStep
        ? [
            {
              "@type": "ListItem",
              position: 3,
              name: activeStep.title,
              item: `${SITE_URL}/docs/${activeStep.slug}`,
            },
          ]
        : []),
    ],
  };

  useSeo({
    title,
    description,
    path,
    jsonLd: breadcrumbsJsonLd,
  });

  useEffect(() => {
    if (slug) {
      const element = document.getElementById(slug);
      if (element) {
        element.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [slug]);

  return (
    <div className="pt-16 min-h-screen bg-background text-foreground font-sans selection:bg-primary selection:text-primary-foreground">
      <main className="container mx-auto px-4 py-12 max-w-5xl">
        <header className="mb-10 max-w-3xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-primary">
            Documentation
          </p>
          <h1 className="mt-4 text-4xl md:text-5xl font-bold tracking-tight leading-tight">
            {activeStep ? activeStep.title : "Using the app, step by step"}
          </h1>
          <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
            Sign in, give the app something to read, then ask your question.
            Each step below shows what happens on screen and what the pipeline
            is doing behind it.
          </p>
        </header>

        {/* Quick jump navigation pills for steps */}
        <nav aria-label="Documentation Steps" className="mb-12 flex flex-wrap gap-2">
          <Link
            to="/docs"
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
              !slug
                ? "bg-primary text-primary-foreground border-primary font-medium"
                : "bg-card text-muted-foreground border-border hover:border-primary/50 hover:text-foreground"
            }`}
          >
            All Steps
          </Link>
          {steps.map((step) => (
            <Link
              key={step.slug}
              to={`/docs/${step.slug}`}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                slug === step.slug
                  ? "bg-primary text-primary-foreground border-primary font-medium"
                  : "bg-card text-muted-foreground border-border hover:border-primary/50 hover:text-foreground"
              }`}
            >
              {step.id}. {step.title.split("&")[0].trim()}
            </Link>
          ))}
        </nav>

        <div className="relative border-l border-border ml-4 md:ml-6 space-y-12">
          {steps.map((step) => {
            const isSelected = slug === step.slug;
            return (
              <div key={step.id} className="relative pl-8 md:pl-12">
                {/* Timeline Dot */}
                <div
                  className={`absolute -left-[20px] top-0 flex h-10 w-10 items-center justify-center rounded-full border bg-card shadow-sm ring-4 ring-background transition-colors ${
                    isSelected
                      ? "border-primary text-primary ring-primary/20 font-extrabold"
                      : "border-border text-primary font-bold"
                  }`}
                >
                  <span className="text-sm">{step.id}</span>
                </div>

                {/* Content Card */}
                <div
                  id={step.slug}
                  className={`grid gap-6 md:grid-cols-2 bg-[hsl(var(--muted))] border rounded p-6 transition-all duration-300 ${
                    isSelected
                      ? "border-primary ring-2 ring-primary/20 shadow-md"
                      : "border-border hover:border-primary/50"
                  }`}
                >
                  {/* Text Content */}
                  <div className="flex flex-col justify-center space-y-4">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-md bg-accent text-primary">
                          {step.icon}
                        </div>
                        <h2 className="text-2xl font-bold text-foreground">
                          {step.title}
                        </h2>
                      </div>
                      <Link
                        to={`/docs/${step.slug}`}
                        aria-label={`Direct link to ${step.title}`}
                        className="text-muted-foreground hover:text-primary transition-colors p-1"
                        title="Direct link"
                      >
                        <Link2 className="w-4 h-4" />
                      </Link>
                    </div>
                    <div className="text-muted-foreground leading-relaxed">
                      {step.description}
                    </div>
                  </div>

                  {/* Image Content */}
                  <div className="relative rounded overflow-hidden border border-border bg-muted/30 aspect-video group cursor-pointer">
                    <div className="absolute inset-0 flex items-center justify-center bg-[hsl(var(--muted))] group-hover:bg-transparent transition-all z-10">
                      <span className="sr-only">View Screenshot</span>
                    </div>
                    <img
                      src={
                        step.imagePath ||
                        `https://placehold.co/600x400/0f172a/06b6d4?text=${encodeURIComponent(step.imagePlaceholderText)}`
                      }
                      alt={step.imageAlt}
                      className="w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-105 transition-all duration-500"
                    />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Docs;
