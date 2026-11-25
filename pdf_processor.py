
from pathlib import Path
from pypdf import PdfReader

# Locate the company handbook
pdf_path = Path("documents/employee_handbook.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

# Read PDF
reader = PdfReader(str(pdf_path))

print("Total pages:", len(reader.pages))

# Extract text
full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    full_text += page_text + "\n"

print("\nExtracted text preview:")
print(full_text[:2000])

# Split into overlapping chunks
chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(full_text):
    end = start + chunk_size
    chunk = full_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

print("\nTotal chunks:", len(chunks))

for index, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {index + 1} ---")
    print(chunk)
