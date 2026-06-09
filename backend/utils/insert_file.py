import io
import os

import re
import requests
from typing import Union
from bs4 import BeautifulSoup
import PyPDF2

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.files.uploadedfile import UploadedFile
from utils.helper import _document_base_path
from django.utils.text import get_valid_filename

class DataLoader:
    """
    Django-integrated loader that:
    - Saves source files into MEDIA_ROOT
    - Extracts clean text
    - Returns (text, stored_paths)
    """

    @staticmethod
    def process_input(
        source: Union[str, UploadedFile],
        username: str
    ) -> dict:
        """
        Returns:
        {
            "text": str,
            "source_path": str,
            "text_path": str
        }
        """

        base_path = _document_base_path(username)

        # ---------- PDF Upload ----------
        if isinstance(source, UploadedFile):
            filename = get_valid_filename(source.name)
            pdf_path = f"{base_path}/{filename}"

            default_storage.save(pdf_path, source)

            text = DataLoader._parse_pdf(
                default_storage.path(pdf_path)
            )

            text_path = DataLoader._save_text(base_path, text)

            return {
                "user": username,
                "text": text,
                "source_path": pdf_path,
                "text_path": text_path,
                "filename": filename,   
                "source_type": "pdf",
            }

        # ---------- URL ----------
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            html = DataLoader._fetch_url(source)

            html_path = f"{base_path}/source.html"
            default_storage.save(html_path, ContentFile(html))

            text = DataLoader._extract_text_from_html(html)
            text_path = DataLoader._save_text(base_path, text)

            return {
                "user": username,
                "text": text,
                "name": source,
                "source_path": html_path,
                "text_path": text_path,
                "source_type": "url",
            }

        # ---------- Local PDF Path ----------
        if isinstance(source, str) and source.lower().endswith(".pdf"):
            text = DataLoader._parse_pdf(source)
            text_path = DataLoader._save_text(base_path, text)

            return {
                "user": username,
                "text": text,
                "source_path": source,
                "text_path": text_path,
                "source_type": "pdf",
            }

        # ---------- Raw Text ----------
        text = source if isinstance(source, str) else ""
        text_path = DataLoader._save_text(base_path, text)

        return {
            "user": username,
            "text": text,
            "name": "text",
            "source_path": None,
            "text_path": text_path,
            "source_type": "text",
        }

    @staticmethod
    def _parse_pdf(path: str) -> str:
        text_content = []
        reader = PyPDF2.PdfReader(path)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)

        return DataLoader._clean_text("\n".join(text_content))

    @staticmethod
    def _fetch_url(url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _extract_text_from_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return DataLoader._clean_text(soup.get_text(separator=" "))

    @staticmethod
    def _save_text(base_path: str, text: str) -> str:
        path = f"{base_path}/extracted.txt"
        default_storage.save(path, ContentFile(text))
        return path

    @staticmethod
    def _clean_text(text: str) -> str:
        # Replace single newlines between words with space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        # Collapse multiple newlines into paragraph break
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()
    
    def load(self, path: str) -> str:
        """
        Simple loader to extract text from a given file path.
        Supports PDF and TXT files.
        """
        if path.lower().endswith(".pdf"):
            return self._parse_pdf(path)
        elif path.lower().endswith(".txt"):
            with default_storage.open(path, "rb") as f:
                with io.TextIOWrapper(f, encoding="utf-8") as text_file:
                    return text_file.read()
        else:
            raise ValueError("Unsupported file type for loading.")

_loader = None

def get_loader():
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader