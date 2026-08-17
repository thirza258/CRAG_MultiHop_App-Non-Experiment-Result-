import { useEffect } from "react";

/**
 * Per-route metadata for a client-rendered SPA.
 *
 * Crawlers that execute JavaScript (Googlebot, Bingbot) pick these up after
 * hydration. Crawlers that do not (Facebook, X, LinkedIn, Slack) only ever see
 * the static tags in `index.html`, which describe the landing page — so keep
 * `index.html` in sync with the values used for the "/" route below.
 */

export const SITE_URL = "https://crag.nevatal.tech";
export const SITE_NAME = "CRAG MultiHop RAG";
export const OG_IMAGE = `${SITE_URL}/og-image.png`;
export const REPO_URL =
  "https://github.com/thirza258/CRAG_MultiHop_App-Non-Experiment-Result-";

type SeoOptions = {
  /** Full <title>, including the site suffix. */
  title: string;
  description: string;
  /** Route path, e.g. "/docs". Used for the canonical URL. */
  path: string;
  /** Keep private/auth-gated routes out of the index. */
  noindex?: boolean;
};

const setMeta = (attr: "name" | "property", key: string, content: string) => {
  const selector = `meta[${attr}="${key}"]`;
  let el = document.head.querySelector<HTMLMetaElement>(selector);
  if (!el) {
    el = document.createElement("meta");
    el.setAttribute(attr, key);
    document.head.appendChild(el);
  }
  el.setAttribute("content", content);
};

const setCanonical = (href: string) => {
  let el = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (!el) {
    el = document.createElement("link");
    el.setAttribute("rel", "canonical");
    document.head.appendChild(el);
  }
  el.setAttribute("href", href);
};

export function useSeo({ title, description, path, noindex = false }: SeoOptions) {
  useEffect(() => {
    const url = `${SITE_URL}${path}`;

    document.title = title;
    setMeta("name", "description", description);
    setMeta(
      "name",
      "robots",
      noindex ? "noindex, nofollow" : "index, follow, max-image-preview:large"
    );

    setMeta("property", "og:type", "website");
    setMeta("property", "og:site_name", SITE_NAME);
    setMeta("property", "og:title", title);
    setMeta("property", "og:description", description);
    setMeta("property", "og:url", url);
    setMeta("property", "og:image", OG_IMAGE);

    setMeta("name", "twitter:card", "summary_large_image");
    setMeta("name", "twitter:title", title);
    setMeta("name", "twitter:description", description);
    setMeta("name", "twitter:image", OG_IMAGE);

    setCanonical(url);
  }, [title, description, path, noindex]);
}
