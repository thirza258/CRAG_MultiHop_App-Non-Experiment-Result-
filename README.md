# Corrective + Reranker + RAG 
---

## Quick Start (Docker — Recommended)

The easiest way to run the entire stack is with Docker. All services (backend, worker, frontend, PostgreSQL, Redis, ChromaDB) start with a single command.

### Prerequisites
- **Docker** and **Docker Compose v2** installed on your machine.

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create the backend environment file**
   ```bash
   cp backend/.env.example backend/.env
   ```
   Edit `backend/.env` and add your LLM API keys (`OPENAI_API_KEY`, etc.) if needed.

3. **Create the frontend environment file**
   ```bash
   cp frontend/.env.example frontend/.env
   ```
   The default values work out of the box for Docker.

4. **Start all services**
   ```bash
   docker compose up -d --build
   ```
   The first build takes a few minutes (model downloads, package installs). Subsequent starts are fast.

5. **Access the app**
   Open [http://localhost:5151](http://localhost:5151) in your browser.

### Useful Docker commands

| Command | Description |
|---------|-------------|
| `docker compose up -d --build` | Build and start all services in the background |
| `docker compose logs -f backend` | Follow backend logs |
| `docker compose logs -f worker` | Follow worker logs |
| `docker compose down` | Stop all services |
| `docker compose down -v` | Stop all services and delete volumes (resets DB/ChromaDB) |

### Services and ports

| Service | External Port | Internal Port |
|---------|--------------|---------------|
| Frontend (Nginx) | `5151` | `5176` |
| Backend (Daphne) | `8051` | `8000` |
| ChromaDB | `8002` | `8000` |
| PostgreSQL | `5432` | `5432` |
| Redis | `6351` | `6379` |

---

## Local Development (without Docker)

### Prerequisites
- **Python 3.11+** and **npm** installed.
- **PostgreSQL**, **Redis**, and **ChromaDB** servers running locally.

### Steps

1. **Set up the backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # edit with your local settings
   python manage.py migrate
   python manage.py runserver
   ```

2. **Start Celery worker** (in a second terminal)
   ```bash
   cd backend
   source venv/bin/activate
   celery -A ragreader worker --loglevel=info
   ```

3. **Run the frontend** (in a third terminal)
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Access the app at [http://localhost:5173](http://localhost:5173).

# MultiHop Corrective RAG Pipeline

Implementasi sistem **Retrieval-Augmented Generation (RAG)** untuk tugas akhir yang mengintegrasikan:

* **Dense Retrieval**
* **Sparse Retrieval (BM25)**
* **Hybrid Retrieval**
* **Corrective RAG (CRAG)**
* **Reranking**
* **Multi-Hop Retrieval Orchestration**

Sistem dirancang untuk meningkatkan kualitas retrieval dan mengurangi hallucination pada query kompleks yang membutuhkan **multi-hop reasoning** menggunakan dataset **MultiHop-RAG**.

---

# Overview

Repository ini merupakan implementasi pipeline RAG terintegrasi yang menggabungkan tiga lapisan utama:

1. **Core Retriever (Base RAG)**
   Menangani retrieval dokumen menggunakan:

   * Dense Retrieval berbasis embedding
   * Sparse Retrieval berbasis BM25
   * Hybrid Retrieval

2. **Corrective RAG Wrapper (CRAG)**
   Menambahkan mekanisme:

   * Self-grading terhadap hasil retrieval
   * Corrective retrieval
   * Fallback retrieval ketika konteks dianggap tidak relevan

3. **MultiHop Orchestrator**
   Mengelola proses retrieval multi-tahap untuk query yang membutuhkan reasoning lintas dokumen.

Pipeline ini dievaluasi menggunakan dataset **MultiHop-RAG** untuk mengukur:

* Kualitas retrieval
* Kualitas generation
* Efektivitas corrective retrieval
* Dampak reranking terhadap performa akhir

Repository ini juga mengimplementasikan web untuk pengguna yang ingin mencoba pipeline ini dengan mengsubmit dokumen.

---

# Architecture

```text
User Query
    │
    ▼
MultiHop Orchestrator
    │
    ▼
CRAG Wrapper
    │
    ▼
Core Retriever
 ┌───────────────┬───────────────┐
 │ Dense         │ Sparse BM25   │
 │ Retrieval     │ Retrieval     │
 └───────────────┴───────────────┘
            │
            ▼
      Hybrid Retrieval
            │
            ▼
        Reranker
            │
            ▼
 Context Selection
            │
            ▼
        Generator
            │
            ▼
      Final Answer
```

---

# Features

* Dense Retrieval menggunakan embedding model
* Sparse Retrieval menggunakan BM25
* Hybrid Retrieval
* Corrective Retrieval (CRAG)
* Multi-Hop Query Orchestration
* Reranking menggunakan Jina Reranker
* Evaluation pipeline untuk retrieval dan generation
* Support eksperimen berbagai model retrieval dan generation
* Modular pipeline architecture

---

# Technologies
Teknologi yang digunakan berasal dari implementasi pipeline, lalu model yang digunakan adalah model terbaik berdasarkan percobaan eksperimen
## Retrieval

* Dense Vector Retrieval
* BM25 Sparse Retrieval
* Hybrid Retrieval

## Embedding Model

* `gemini-embedding-2-preview`

## Reranker

* `jina-reranker-v3`

## Generator LLM

* `qwen/qwen3-30b-a3b-instruct-2507`

---

# Dataset

Menggunakan dataset:

* **MultiHop-RAG**

Dataset ini berisi:

* Multi-hop queries
* Ground truth answers
* Supporting evidence
* Real-world news articles

Dataset digunakan untuk mengevaluasi kemampuan sistem dalam:

* Retrieval lintas dokumen
* Multi-step reasoning
* Hallucination mitigation

---

# Evaluation Metrics (TA)

Retrieval Metrics
- Accuracy (Hit Rate)
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
Generation Metrics
- BERTScore
- RAGAs
- Faithfulness
- Answer Relevancy
- Answer Correctness 

# Evaluation Metrics (Web App)

  * Faithfulness
  * Answer Relevancy

---

# Project Structure

```text
.
├── backend/
│   ├── retrieval/
│   ├── reranker/
│   ├── crag/
│   ├── multihop/
│   ├── generation/
│   └── evaluation/
│
├── frontend/
│
├── experiments/
│
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

---

# Environment Variables

Create `.env` file:

```env
OPENROUTER_API_KEY=
```

---

# Running the Project


## Docker

```bash
docker compose up -d --build
```

---

# Experiment Configuration

Eksperimen dapat dilakukan dengan mengubah konfigurasi:

* Dense embedding model
* Sparse retrieval method
* Reranker model
* LLM generator
* Retrieval top-k
* Corrective retrieval threshold

---

# Research Objective

Penelitian ini bertujuan untuk:

* Meningkatkan kualitas retrieval pada query multi-hop
* Mengurangi hallucination pada sistem RAG
* Mengevaluasi efektivitas corrective retrieval
* Mengukur pengaruh reranker terhadap retrieval relevance
* Membandingkan Dense, Sparse, dan Hybrid Retrieval

---

# References

* Yan et al. — *Corrective Retrieval-Augmented Generation*
* Tang et al. — *MultiHop-RAG Benchmark*
* RAGAs Evaluation Framework
* BERTScore

---

# License

This project is developed for academic and research purposes.

