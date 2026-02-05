# Langraph Library

[![PyPI Version](https://img.shields.io/pypi/v/langraph_library.svg)](https://pypi.org/project/langraph_library) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Build Status](https://img.shields.io/github/actions/workflow/status/WalidRAMTANI/langraph_library/ci.yml?branch=main)](https://github.com/WalidRAMTANI/langraph_library/actions)

A compact, extensible library to build, manipulate, and query language-aware graphs — "Langraphs". Langraphs connect language artifacts (texts, embeddings, models, prompts) and provide utilities to construct graph-based context, perform semantic search, traverse relationships, and integrate with LLMs and vector stores.

Why Langraph?
- Model language artifacts and relationships explicitly (documents, prompts, embeddings, sources).
- Build semantic graphs for retrieval-augmented generation (RAG), knowledge graphs, and provenance tracking.
- Lightweight API that integrates with vector stores and common LLM tooling.

Features
- Graph primitives: nodes, edges, typed properties
- Helpers to index texts and embeddings
- Query and traversal utilities (neighbors, depth-limited traversal, semantic nearest neighbors)
- Import/export (JSON, GraphML, DOT)
- Integrations (pluggable vector store adapters and LLM connectors)
- Optional async-friendly operations for IO-bound tasks

Installation

From PyPI (recommended)
```
pip install langraph_library
```

From source
```
git clone https://github.com/WalidRAMTANI/langraph_library.git
cd langraph_library
pip install -e .
```

Quick start (Python)

This is a minimal, illustrative example. Adjust names and calls to match the library API.

```python
from langraph import Langraph, Node, Edge
# Create a graph
g = Langraph()

# Add nodes
doc = Node(id="doc:1", type="document", text="The mitochondria is the powerhouse of the cell.")
emb_node = Node(id="emb:1", type="embedding", vector=[0.01, 0.5, ...])

g.add_node(doc)
g.add_node(emb_node)

# Connect nodes
g.add_edge(Edge(source="doc:1", target="emb:1", relation="has_embedding"))

# Query neighbors
neighbors = g.neighbors("doc:1", depth=1)
print(neighbors)

# Semantic search (requires vector-store adapter)
results = g.semantic_search(query="cell energy organelle", top_k=5)
for r in results:
    print(r.node_id, r.score)
```

Core concepts
- Node: a graph vertex representing an artifact (document, embedding, prompt, model, person).
- Edge: a typed relationship between nodes (e.g., `references`, `has_embedding`, `derived_from`).
- Adapter: interface to pluggable systems (vector dbs, LLMs, storage).
- Index: built-in helpers to index documents into a chosen vector store and link them to nodes.

API overview (high-level)
- Langraph(): core graph container
  - add_node(node: Node)
  - add_edge(edge: Edge)
  - get_node(node_id) -> Node
  - neighbors(node_id, depth=1) -> List[Node]
  - traverse(start_id, strategy="bfs", depth=None)
  - semantic_search(query, top_k=10, adapter=None)
- Node(type: str, id: str, **metadata)
- Edge(source: str, target: str, relation: str, **metadata)

Configuration
- Environment variables:
  - LANGRAPH_VECTOR_ADAPTER: default adapter name
  - LANGRAPH_DEFAULT_INDEX: default index name
- Programmatic configuration via a Config object (see docs).

Vector store & LLM integration
Langraph supports a pluggable adapter model for vector stores (FAISS, Milvus, Pinecone, etc.) and LLM providers. Provide credentials and adapter-specific config via environment variables or a config file. Example adapter registration:

```python
from langraph.adapters import register_adapter, FaissAdapter
register_adapter("faiss", FaissAdapter(index_path="/data/index"))
```

CLI
(If your project provides a CLI, document commands here. Example:)
```
langraph init    # initialize a graph project
langraph index   # index documents
langraph query   # run a query
```

Examples & tutorials
- examples/basic_usage.py — A short end-to-end demo: build graph, index docs, query.
- examples/rag_pipeline.py — RAG pipeline using a vector store adapter and an LLM adapter.
(Replace with actual example paths in the repo.)

Development

Set up dev environment
```
git clone https://github.com/WalidRAMTANI/langraph_library.git
cd langraph_library
python -m venv .venv
source .venv/bin/activate
pip install -r dev-requirements.txt
pip install -e .
```

Run tests
```
pytest tests
```

Code style
- Black for formatting
- Flake8/ruff for linting
- Type hints encouraged; run mypy for type checks

Contributing
Contributions are welcome! Please:
1. Open an issue to discuss major changes.
2. Create a branch from `main` with a descriptive name.
3. Submit a PR with tests and documentation updates.

License
This project is licensed under the MIT License — see the LICENSE file for details.

Roadmap (suggested)
- More adapter implementations (Milvus, Pinecone, Weaviate)
- Graph visualization utilities (SVG/DOT exports)
- Additional provenance and lineage helpers
- Improved async batch indexing

Acknowledgments
- Inspired by retrieval-augmented generation and graph-based knowledge modeling patterns.

Contact
For questions or help, open an issue or reach out via GitHub: [WalidRAMTANI](https://github.com/WalidRAMTANI).
