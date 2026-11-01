
import {
  Globe,
  UserPlus,
  FileText,
  Loader2,
  MessageSquare,
  Search,
  Activity,
  BarChart,
  Settings,
  CheckCircle,
  Database,
} from "lucide-react";
import { DocStep } from "../../types/types";

import landingImage from "../../assets/docs/docs_landingpage.png";
import loginImage from "../../assets/docs/docs_login.png";
import pdfInputImage from "../../assets/docs/docs_pdfinput.png";
import initImage from "../../assets/docs/docs_init.png";
import chatInterfaceImage from "../../assets/docs/docs_chat.png";
import deepAnalysisImage from "../../assets/docs/docs_deepanalysis.png";
import groundTruthImage from "../../assets/docs/docs_groundtruth.png";
import hybridSettingsImage from "../../assets/docs/docs_hybrid.png";
import scoringImage from "../../assets/docs/docs_scoring.png";
import sidebarImage from "../../assets/docs/docs_sidebar.png";
import evaluationImage from "../../assets/docs/docs_evaluation.png";

export const steps: DocStep[] = [
  {
    id: 1,
    slug: "landing-page",
    title: "Platform Overview & Landing Page",
    description:
      "Explore the homepage to learn about the Corrective Multi-Hop RAG architecture, test the interactive trace simulator, and start chatting or reading documentation.",
    icon: <Globe className="w-6 h-6 text-primary" />,
    imagePath: landingImage,
    imageAlt: "CRAG MultiHop RAG Landing Page and Live Simulator",
    imagePlaceholderText: "Landing Page Image",
  },
  {
    id: 2,
    slug: "account-creation",
    title: "Account Creation & Login",
    description:
      "Create a username and email. This unified login allows you to resume sessions seamlessly across devices.",
    icon: <UserPlus className="w-6 h-6 text-primary" />,
    imagePath: loginImage,
    imageAlt: "Login Screen Interface",
    imagePlaceholderText: "Login Image",
  },
  {
    id: 3,
    slug: "document-ingestion",
    title: "Document Ingestion",
    description:
      "Insert your PDF URL or paste text directly. The system prepares your data for the knowledge base.",
    icon: <FileText className="w-6 h-6 text-primary" />,
    imagePath: pdfInputImage,
    imageAlt: "PDF Upload Interface",
    imagePlaceholderText: "PDF / URL Input Image",
  },
  {
    id: 4,
    slug: "initialization",
    title: "Initialization",
    description:
      "The system initializes your PDF. A loading indicator signifies the processing and vectorization phase.",
    icon: <Loader2 className="w-6 h-6 text-primary animate-spin" />,
    imagePath: initImage,
    imageAlt: "Loading State",
    imagePlaceholderText: "Loading / Initialization Image",
  },
  {
    id: 5,
    slug: "chat-interaction",
    title: "Chat Interaction",
    description: (
      <>
        Engage with the chatbot. By default, it utilizes{" "}
        <span className="text-primary font-bold">Dense RAG</span> to retrieve
        relevant context for your answers.
      </>
    ),
    icon: <MessageSquare className="w-6 h-6 text-primary" />,
    imagePath: chatInterfaceImage,
    imageAlt: "Chat Interface",
    imagePlaceholderText: "Chatbot Interface Image",
  },
  {
    id: 6,
    slug: "deep-analysis",
    title: "Trigger Deep Analysis",
    description:
      "Inspect detailed comparative analysis across retrieval pipelines, chunking configurations, and per-retrieval scores.",
    icon: <Search className="w-6 h-6 text-primary" />,
    imagePath: deepAnalysisImage,
    imageAlt: "Deep Analysis Configuration and Pipeline Comparison",
    imagePlaceholderText: "Close Up: Deep Analysis Button",
  },
  {
    id: 7,
    slug: "ground-truth",
    title: "Select Ground Truth Chunks and Provide Response",
    description:
      "In the Ground Truth view, inspect retrieved text chunks to verify factual accuracy and match against reference ground truth sources.",
    icon: <Database className="w-6 h-6 text-primary" />,
    imagePath: groundTruthImage,
    imageAlt: "Ground Truth Selection and Retrieved Context UI",
    imagePlaceholderText: "Ground Truth Page Image",
  },
  {
    id: 8,
    slug: "hybrid-processing",
    title: "Hybrid Processing",
    description: (
      <>
        The system performs a background task executing{" "}
        <span className="text-secondary font-bold">
          Dense Sparse Hybrid search + Reranking
        </span>{" "}
        to refine accuracy.
      </>
    ),
    icon: <Activity className="w-6 h-6 text-primary" />,
    imagePath: hybridSettingsImage,
    imageAlt: "Retrieval settings with hybrid search enabled",
    imagePlaceholderText: "Processing: Hybrid + Reranker",
  },
  {
    id: 9,
    slug: "scoring-results",
    title: "Scoring & Results",
    description:
      "Receive real-time answer relevancy and faithfulness scores indicating how well the retrieved chunks match your specific query.",
    icon: <BarChart className="w-6 h-6 text-primary" />,
    imagePath: scoringImage,
    imageAlt: "Answer with answer relevancy and faithfulness scores",
    imagePlaceholderText: "Relevance Score Display",
  },
  {
    id: 10,
    slug: "sidebar-configuration",
    title: "Sidebar Configuration",
    description:
      "Use the sidebar to configure advanced settings: toggle multi-hop reasoning, corrective CRAG grading, cross-encoder rerankers, and custom model endpoints.",
    icon: <Settings className="w-6 h-6 text-primary" />,
    imagePath: sidebarImage,
    imageAlt: "Pipeline configuration in the sidebar",
    imagePlaceholderText: "Sidebar Configuration Image",
  },
  {
    id: 11,
    slug: "ground-truth-selection",
    title: "Ground Truth Selection",
    description:
      "In the Ground Truth view, select the specific text chunk you believe represents the accurate answer.",
    icon: <Database className="w-6 h-6 text-primary" />,
    imagePath: groundTruthImage,
    imageAlt: "Ground Truth Selection UI",
    imagePlaceholderText: "Ground Truth Page Image",
  },
  {
    id: 12,
    slug: "evaluation",
    title: "Evaluation",
    description:
      "Once Ground Truth is established, the evaluation metrics specific to that truth will appear for analysis.",
    icon: <CheckCircle className="w-6 h-6 text-primary" />,
    imagePath: evaluationImage,
    imageAlt: "Evaluation metrics shown under an answer",
    imagePlaceholderText: "Evaluation Results Image",
  },
];
