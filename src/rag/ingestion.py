from pathlib import Path
from typing import List
from dataclasses import dataclass


@dataclass
class Document:
    content: str
    metadata: dict
    source: str


class DocumentLoader:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if path.suffix == ".txt":
            return self._load_txt(path)
        elif path.suffix == ".pdf":
            return self._load_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

    def _load_txt(self, path: Path) -> List[Document]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = self._split_text(content)
        return [
            Document(
                content=chunk,
                metadata={"source": str(path), "chunk_index": i},
                source=str(path),
            )
            for i, chunk in enumerate(chunks)
        ]

    def _load_pdf(self, path: Path) -> List[Document]:
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            content = ""
            for page in doc:
                content += page.get_text()
            doc.close()
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

        chunks = self._split_text(content)
        return [
            Document(
                content=chunk,
                metadata={"source": str(path), "chunk_index": i},
                source=str(path),
            )
            for i, chunk in enumerate(chunks)
        ]

    def load_bytes(self, filename: str, content: bytes, source: str = "") -> List[Document]:
        """Load a document from in-memory bytes (e.g. an uploaded file)."""
        path = Path(filename)
        if not source:
            source = filename
        if path.suffix == ".txt":
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("TXT file must be UTF-8 encoded")
            chunks = self._split_text(text)
        elif path.suffix == ".pdf":
            try:
                import pymupdf
                import io
                doc = pymupdf.open(stream=content, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
            except ImportError:
                raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
            chunks = self._split_text(text)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return [
            Document(
                content=chunk,
                metadata={"source": source, "chunk_index": i},
                source=source,
            )
            for i, chunk in enumerate(chunks)
        ]

    def _split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    def load_directory(self, dir_path: str) -> List[Document]:
        path = Path(dir_path)
        all_docs = []
        for file_path in path.glob("**/*"):
            if file_path.suffix in [".txt", ".pdf"]:
                all_docs.extend(self.load_file(str(file_path)))
        return all_docs
