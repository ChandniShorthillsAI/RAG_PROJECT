import os
import time
import re
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RagPipeline:
    def __init__(self, txt_file_path="data/human_evolution_combined.txt", index_path="embeddings/faiss_index", chunk_size=500, chunk_overlap=50):
        self.txt_file_path = txt_file_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if os.path.exists(index_path):
            logging.info(f"🧹 Removing existing FAISS index at {index_path}")
            for file in os.listdir(index_path):
                file_path = os.path.join(index_path, file)
                os.remove(file_path)
        else:
            os.makedirs(index_path, exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_and_chunk_text(self) -> List[Document]:
        logging.info(f"📄 Reading file: {self.txt_file_path}")
        with open(self.txt_file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        cleaned_text = self.clean_text(raw_text)
        chunks = self.splitter.split_text(cleaned_text)

        documents = [
            Document(
                page_content=chunk,
                metadata={"source": self.txt_file_path}
            )
            for chunk in chunks
        ]

        logging.info(f"✅ Created {len(documents)} chunks from text")
        return documents

    def embed_and_store(self, chunks: List[Document]):
        logging.info("🔍 Starting embedding...")

        start = time.time()

        # Wrap embedding in tqdm progress bar
        class tqdmEmbedding:
            def __init__(self, embedder, total):
                self.embedder = embedder
                self.pbar = tqdm(total=total, desc="💡 Embedding chunks")

            def embed_documents(self, texts):
                embeddings = []
                for text in texts:
                    embeddings.append(self.embedder.embed_documents([text])[0])
                    self.pbar.update(1)
                self.pbar.close()
                return embeddings

        embedder = tqdmEmbedding(self.embedding_model, total=len(chunks))
        db = FAISS.from_documents(chunks, embedder)
        db.save_local(self.index_path)

        end = time.time()
        logging.info(f"✅ Embedding complete in {round(end - start, 2)} seconds.")
        logging.info(f"📦 FAISS index saved at {self.index_path}")

    def run(self):
        chunks = self.load_and_chunk_text()
        self.embed_and_store(chunks)

if __name__ == "__main__":
    pipeline = RagPipeline()
    pipeline.run()
    logging.info("🎉 RAG pipeline completed successfully.")
