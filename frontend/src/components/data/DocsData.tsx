
import {
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

import loginImage from "../../assets/docs/docs_login.png";
import pdfInputImage from "../../assets/docs/docs_pdfinput.png";
import initImage from "../../assets/docs/docs_init.png";
import chatInterfaceImage from "../../assets/docs/docs_chat.png";
import deepAnalysisImage from "../../assets/docs/docs_deepanalysis.png";

export const steps: DocStep[] = [
  {
    id: 1,
    title: "Account Creation & Login",
    description:
      "Create a username and email. This unified login allows you to resume sessions seamlessly across devices.",
    icon: <UserPlus className="w-6 h-6 text-primary" />,
    imagePath: loginImage,
    imageAlt: "Login Screen Interface",
    imagePlaceholderText: "Login Image",
  },
  {
    id: 2,
    title: "Document Ingestion",
    description:
      "Insert your PDF URL or paste text directly. The system prepares your data for the knowledge base.",
    icon: <FileText className="w-6 h-6 text-primary" />,
    imagePath: pdfInputImage,
    imageAlt: "PDF Upload Interface",
    imagePlaceholderText: "PDF / URL Input Image",
  },
  {
    id: 3,
    title: "Initialization",
    description:
      "The system initializes your PDF. A loading indicator signifies the processing and vectorization phase.",
    icon: <Loader2 className="w-6 h-6 text-primary animate-spin" />,
    imagePath: initImage,
    imageAlt: "Loading State",
    imagePlaceholderText: "Loading / Initialization Image",
  },
  {
    id: 4,
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
    id: 5,
    title: "Trigger Deep Analysis",
    description:
      "Click the 'Deep Analysis' button located under the chatbot's response to investigate the source validity.",
    icon: <Search className="w-6 h-6 text-primary" />,
    imageAlt: "Close up of Deep Analysis Button",
    imagePlaceholderText: "Close Up: Deep Analysis Button",
  },
  {
    id: 6,
    title: "Select Ground Truth Chunks and Provide Response",
    description:
      "In the Ground Truth view, select the specific text chunk you believe represents the accurate answer and provide your expected response.",
    icon: <Database className="w-6 h-6 text-primary" />,
    imageAlt: "Ground Truth Selection UI",
    imagePlaceholderText: "Ground Truth Page Image",
  },
  {
    id: 7,
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
    imageAlt: "Background Process Diagram",
    imagePlaceholderText: "Processing: Hybrid + Reranker",
  },
  {
    id: 8,
    title: "Scoring & Results",
    description:
      "Receive a relevance score indicating how well the retrieved chunks match your specific query.",
    icon: <BarChart className="w-6 h-6 text-primary" />,
    imagePath: deepAnalysisImage,
    imageAlt: "Scoring Result UI",
    imagePlaceholderText: "Relevance Score Display",
  },
  {
    id: 9,
    title: "Sidebar Configuration",
    description:
      "Use the sidebar to configure advanced settings or prepare to add Ground Truth data for evaluation.",
    icon: <Settings className="w-6 h-6 text-primary" />,
    imageAlt: "Sidebar Settings",
    imagePlaceholderText: "Sidebar Configuration Image",
  },
  {
    id: 10,
    title: "Ground Truth Selection",
    description:
      "In the Ground Truth view, select the specific text chunk you believe represents the accurate answer.",
    icon: <Database className="w-6 h-6 text-primary" />,
    imageAlt: "Ground Truth Selection UI",
    imagePlaceholderText: "Ground Truth Page Image",
  },
  {
    id: 11,
    title: "Evaluation",
    description:
      "Once Ground Truth is established, the evaluation metrics specific to that truth will appear for analysis.",
    icon: <CheckCircle className="w-6 h-6 text-primary" />,
    imageAlt: "Evaluation Metrics",
    imagePlaceholderText: "Evaluation Results Image",
  },
];
