"""
Text Extraction Module for AI-Generated Text Detector (SPARK).
Handles:
- Document extraction: PDF (.pdf), Word (.docx), Plain Text (.txt, .md, .rtf, .csv)
- Web extraction: Article and blog URLs (BeautifulSoup + robust stdlib fallback)
- Text sanitization and validation
- Zero mandatory external top-level dependencies for bulletproof import resilience
"""

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Dict, Any, Tuple, Optional


class SimpleHTMLTextExtractor(HTMLParser):
    """Zero-dependency HTML body parser from Python standard library."""
    def __init__(self):
        super().__init__()
        self.texts = []
        self.title = ""
        self.in_title = False
        self.ignore = False
        self.ignore_tags = {'script', 'style', 'noscript', 'header', 'footer', 'nav', 'aside', 'svg'}

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()
        if tag_lower == 'title':
            self.in_title = True
        elif tag_lower in self.ignore_tags:
            self.ignore = True

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        if tag_lower == 'title':
            self.in_title = False
        elif tag_lower in self.ignore_tags:
            self.ignore = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data.strip()
        elif not self.ignore:
            cleaned = data.strip()
            if cleaned:
                self.texts.append(cleaned)

    def get_text(self) -> str:
        return ' '.join(self.texts)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-printable characters."""
    if not text:
        return ""
    # Normalize unicode spaces & tabs
    text = re.sub(r'[\r\t\f\v ]+', ' ', text)
    # Normalize multiple newlines (keep paragraph breaks)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Strip leading/trailing whitespaces
    return text.strip()


def extract_from_docx(file_bytes: bytes) -> str:
    """Extract text from Word .docx file using docx library or fallback XML parsing."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text.strip())
        return "\n\n".join(paragraphs)
    except Exception:
        # Robust fallback using built-in zipfile and XML
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                xml_content = zf.read('word/document.xml')
                tree = ET.fromstring(xml_content)
                namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                paragraphs = []
                for p in tree.iterfind('.//w:p', namespaces):
                    texts = [node.text for node in p.iterfind('.//w:t', namespaces) if node.text]
                    if texts:
                        paragraphs.append(''.join(texts))
                return "\n\n".join(paragraphs)
        except Exception as e:
            raise ValueError(f"Failed to parse Word document: {str(e)}")


def extract_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_str = page.extract_text() or ""
            if page_str.strip():
                pages_text.append(page_str.strip())
        if not pages_text:
            raise ValueError("No extractable text found in PDF (it may contain scanned images).")
        return "\n\n".join(pages_text)
    except ImportError:
        raise ValueError("PDF parsing requires the 'pypdf' library. Please install requirements.txt or paste text directly.")
    except Exception as e:
        raise ValueError(f"Failed to parse PDF document: {str(e)}")


def extract_from_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract text from uploaded document.
    Supports .pdf, .docx, .txt, .md, .rtf, .csv
    """
    filename_lower = filename.lower()
    raw_text = ""
    file_type = "unknown"

    if filename_lower.endswith('.pdf'):
        file_type = "PDF Document"
        raw_text = extract_from_pdf(file_bytes)
    elif filename_lower.endswith('.docx'):
        file_type = "Word Document (.docx)"
        raw_text = extract_from_docx(file_bytes)
    elif filename_lower.endswith(('.txt', '.md', '.rtf', '.csv', '.text')):
        file_type = "Plain Text / Markdown"
        for encoding in ['utf-8', 'latin-1', 'utf-16', 'cp1252']:
            try:
                raw_text = file_bytes.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if not raw_text:
            raw_text = file_bytes.decode('utf-8', errors='ignore')
    else:
        # Attempt plain text decoding as fallback
        try:
            raw_text = file_bytes.decode('utf-8')
            file_type = "Generic Text"
        except Exception:
            raise ValueError(f"Unsupported file format: '{filename}'. Please upload PDF, Word (.docx), or text files.")

    cleaned = clean_text(raw_text)
    words = cleaned.split()
    
    return {
        "filename": filename,
        "file_type": file_type,
        "raw_text": cleaned,
        "word_count": len(words),
        "char_count": len(cleaned),
        "preview": cleaned[:300] + ("..." if len(cleaned) > 300 else "")
    }


def extract_from_url(url: str) -> Dict[str, Any]:
    """
    Extract main article/blog content from a web URL.
    Uses BeautifulSoup if available, with pure stdlib fallback if missing.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Attempt fetch and parse using BeautifulSoup + requests
    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find('h1'):
            title = soup.find('h1').get_text().strip()
        else:
            title = url

        # Remove non-content elements
        for element in soup(['script', 'style', 'noscript', 'nav', 'header', 'footer', 
                             'aside', 'form', 'svg', 'figure', 'button', 'iframe']):
            element.decompose()

        # Look for primary content containers
        content_node = None
        for tag in ['article', 'main', 'div[role="main"]', '.post-content', '.article-body', '.entry-content']:
            found = soup.select_one(tag)
            if found and len(found.get_text().split()) > 40:
                content_node = found
                break

        if not content_node:
            content_node = soup.body if soup.body else soup

        # Extract all paragraph texts
        paragraphs = []
        for p in content_node.find_all(['p', 'h2', 'h3', 'h4', 'blockquote']):
            txt = p.get_text().strip()
            if len(txt.split()) >= 5:
                paragraphs.append(txt)

        if not paragraphs:
            text = content_node.get_text(separator=' ')
            cleaned = clean_text(text)
        else:
            cleaned = clean_text("\n\n".join(paragraphs))

    except ImportError:
        # Robust zero-dependency fallback using urllib and standard library HTMLParser
        try:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as response:
                html_bytes = response.read()
                html_text = html_bytes.decode('utf-8', errors='ignore')

            parser = SimpleHTMLTextExtractor()
            parser.feed(html_text)
            title = parser.title or url
            cleaned = clean_text(parser.get_text())
        except Exception as e:
            raise ValueError(f"Unable to extract text from URL '{url}': {str(e)}")
    except Exception as e:
        raise ValueError(f"Unable to fetch URL '{url}': {str(e)}")

    words = cleaned.split()

    return {
        "url": url,
        "title": title,
        "raw_text": cleaned,
        "word_count": len(words),
        "char_count": len(cleaned),
        "preview": cleaned[:300] + ("..." if len(cleaned) > 300 else "")
    }
