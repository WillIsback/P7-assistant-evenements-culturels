"""Test de fumée de l'environnement : les briques clés s'importent et s'instancient.

Aucun appel réseau : les clients LLM/embeddings sont construits sans être interrogés,
et l'index Faiss est construit avec des embeddings factices.
"""

from __future__ import annotations

import faiss
import numpy as np
from fastapi import FastAPI
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr


def test_faiss_indexe_et_recherche() -> None:
    vecteurs = np.random.default_rng(0).random((10, 8), dtype=np.float32)
    index = faiss.IndexFlatL2(8)
    index.add(vecteurs)
    _, voisins = index.search(vecteurs[:1], 1)
    assert index.ntotal == 10
    assert voisins[0][0] == 0


def test_vectorstore_faiss_langchain() -> None:
    store = FAISS.from_texts(["concert de jazz", "exposition photo"], FakeEmbeddings(size=8))
    assert store.index.ntotal == 2
    assert len(store.similarity_search("jazz", k=1)) == 1


def test_clients_compatibles_openai_sans_reseau() -> None:
    cle = SecretStr("cle-test")
    chat = ChatOpenAI(model="modele-test", api_key=cle, base_url="http://localhost/v1")
    embeddings = OpenAIEmbeddings(model="modele-test", api_key=cle, base_url="http://localhost/v1")
    assert chat.model_name == "modele-test"
    assert embeddings.model == "modele-test"


def test_fastapi() -> None:
    assert FastAPI().title == "FastAPI"
