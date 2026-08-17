import React from "react";
import { steps } from "../components/data/DocsData";
import { useSeo } from "../lib/seo";
import Footer from "../components/Footer";


const Docs: React.FC = () => {
  useSeo({
    title: "How to use the app, step by step | CRAG MultiHop RAG",
    description:
      "A walkthrough of the app: signing in, adding a PDF, URL or pasted text, waiting for indexing, chatting with the corrective multi-hop pipeline, and reading the retrieved chunks and evaluation scores.",
    path: "/docs",
  });

  return (
    <div className="pt-16 min-h-screen bg-background text-foreground font-sans selection:bg-primary selection:text-primary-foreground">
      <main className="container mx-auto px-4 py-12 max-w-5xl">
        <header className="mb-12 max-w-3xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-primary">
            Documentation
          </p>
          <h1 className="mt-4 text-4xl md:text-5xl font-bold tracking-tight leading-tight">
            Using the app, step by step
          </h1>
          <p className="mt-6 text-lg leading-relaxed text-muted-foreground">
            Sign in, give the app something to read, then ask your question.
            Each step below shows what happens on screen and what the pipeline
            is doing behind it.
          </p>
        </header>

        <div className="relative border-l border-border ml-4 md:ml-6 space-y-12">
          {steps.map((step) => (
            <div key={step.id} className="relative pl-8 md:pl-12">
              {/* Timeline Dot */}
              <div className="absolute -left-[20px] top-0 flex h-10 w-10 items-center justify-center rounded-full border border-border bg-card shadow-sm ring-4 ring-background">
                <span className="text-primary font-bold text-sm">
                  {step.id}
                </span>
              </div>

              {/* Content Card */}
              <div className="grid gap-6 md:grid-cols-2 bg-card/50 border border-border rounded-lg p-6 hover:border-primary/50 transition-colors duration-300">
                {/* Text Content */}
                <div className="flex flex-col justify-center space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-md bg-accent text-primary">
                      {step.icon}
                    </div>
                    <h2 className="text-2xl font-bold text-foreground">
                      {step.title}
                    </h2>
                  </div>
                  <p className="text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>

                {/* Image Content */}
                <div className="relative rounded-lg overflow-hidden border border-border bg-muted/30 aspect-video group cursor-pointer">
                  <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-transparent transition-all z-10">
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
          ))}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Docs;
