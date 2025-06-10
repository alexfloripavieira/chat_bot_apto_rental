import os
import sys
from pathlib import Path

import pytest
from langchain_core.documents.base import Document

# Ensure project root is on the path so ``vectorstore`` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vectorstore


def test_load_documents_moves_and_returns(tmp_path, monkeypatch):
    # Create temporary rag directory and sample files
    rag_dir = tmp_path
    file1 = rag_dir / "doc1.txt"
    file2 = rag_dir / "doc2.txt"
    file1.write_text("hello one")
    file2.write_text("hello two")

    # Patch the directory path used by vectorstore
    monkeypatch.setattr(vectorstore, "RAG_FILES_DIR", str(rag_dir))

    # Call load_documents
    docs = vectorstore.load_documents()

    processed = rag_dir / "processed"
    # Verify processed directory exists and files moved
    assert processed.exists(), "processed directory should be created"
    moved_files = sorted(p.name for p in processed.iterdir())
    assert moved_files == ["doc1.txt", "doc2.txt"]

    # Original files should no longer exist in rag_dir
    assert not file1.exists() and not file2.exists()

    # Returned objects are Document instances
    assert docs and all(isinstance(d, Document) for d in docs)
