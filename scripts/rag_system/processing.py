# scripts/rag_system/processing.py

import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from typing import List
import re

# --- Local Imports ---
import config

# --- Library Imports ---
from langchain_core.documents import Document
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder

from markdownify import markdownify as md

# --- Helper Functions ---

def _structurally_split_markdown(markdown_text: str) -> List[str]:
    """
    Splits a markdown document into large sections based on top-level headings (#).
    This helps to keep document structure intact before semantic chunking.
    """
    # First, strip out useless image links
    markdown_text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_text)
    # Split the text by lines that start with '# ' (a top-level heading)
    # The (?m) flag enables multi-line mode, so '^' matches the start of each line
    sections = re.split(r'(?m)^# ', markdown_text)
    
    # The first split part is usually empty or a preamble, so we re-add the '#'
    # to all subsequent parts and filter out any empty strings.
    return ["# " + section.strip() for section in sections if section.strip()]

def _convert_html_to_markdown(html_text: str) -> str:
    """
    Uses the markdownify library to convert a string containing HTML
    (especially tables) into clean, well-formatted Markdown.
    """
    # The `md()` function does the conversion.
    # heading_style="ATX" ensures that <h1> becomes #, <h2> becomes ##, etc.
    return md(html_text, heading_style="ATX")


def _run_mineru_cli_and_save(pdf_path: Path) -> str:

    # Use shutil.which to find the full path to the mineru executable.
    # This is robust and respects the activated Python/Conda environment.
    mineru_executable = shutil.which("mineru")
    if not mineru_executable:
        raise FileNotFoundError("Could not find 'mineru' command.")
    
    temp_output_dir = config.PROJECT_ROOT / f"mineru_temp_output_{pdf_path.stem}"
    if temp_output_dir.exists(): 
        shutil.rmtree(temp_output_dir)
    temp_output_dir.mkdir()
    
    # Construct the command as a list of strings for security and correctness.
    command = [
        mineru_executable,
        "--path", str(pdf_path),
        "--output", str(temp_output_dir),
        "--lang", "latin",
        "--backend", "pipeline",
        "--method","ocr"
    ]

    try:
        tqdm.write(f"  -> Running MinerU CLI for {pdf_path.name}...")
        
        subprocess.run(
            command,
            check=True,          # Raise an exception if the command returns a non-zero (error) code.
            capture_output=True, # Capture stdout and stderr streams.
            text=True            # Decode stdout/stderr from bytes into strings.
        )

        nested_output_path = temp_output_dir / pdf_path.stem / "ocr"
        md_file = next(nested_output_path.glob("*.md"), None)
        json_file = next(nested_output_path.glob("*_model.json"), None)



        if md_file:
            shutil.move(str(md_file), config.MARKDOWN_DEST_DIR / md_file.name)
            tqdm.write(f"     -> Saved {md_file.name} to {config.MARKDOWN_DEST_DIR.name}/")
        if json_file:
            shutil.move(str(json_file), config.JSON_DEST_DIR / json_file.name)
            tqdm.write(f"     -> Saved {json_file.name} to {config.JSON_DEST_DIR.name}/")

    except subprocess.CalledProcessError as e:
        tqdm.write(f"\n❌ MinerU failed on {pdf_path.name}.\n   Error: {e.stderr}\n")
    finally:
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir)


def _chunk_text(text: str, source_name: str, chunker_encoder: HuggingFaceEncoder) -> List[Document]:
    """Chunks a single block of text into LangChain Documents."""
    if not text:
        return []

    chunker = StatisticalChunker(
        encoder=chunker_encoder,
        min_split_tokens=config.CHUNK_MIN_TOKENS,
        max_split_tokens=config.CHUNK_MAX_TOKENS
    )
    chunks = chunker([text])[0]
    return [
        Document(
            page_content=chunk_content,
            metadata={"source": source_name, "chunk_number": i + 1}
        ) for i, chunk_content in enumerate([chunk.content for chunk in chunks])
    ]

def _chunk_document_holistically(full_text: str, source_name: str, chunker_encoder) -> List[Document]:
    """
    The definitive chunking strategy. It groups lines by type, then intelligently
    merges orphaned headings with their subsequent tables.
    """
    docs = []
    lines = full_text.splitlines()
    blocks, current_block, current_type = [], "", None

    # Step 1: Group lines into raw blocks of "prose" or "table"
    for line in lines:
        line_strip = line.strip()
        if not line_strip: continue

        line_type = "table" if line_strip.startswith('|') and line_strip.endswith('|') else "prose"
        
        if line_type != current_type and current_block:
            blocks.append((current_type, current_block.strip()))
            current_block = ""
        
        current_block += line + "\n"
        current_type = line_type
    
    if current_block.strip():
        blocks.append((current_type, current_block.strip()))

    # Step 2: Post-process the blocks to merge orphaned headings with their tables
    merged_blocks = []
    i = 0
    while i < len(blocks):
        current_type, current_content = blocks[i]
        # Check if the current block is a heading-only prose block
        if current_type == "prose" and current_content.startswith('#') and '\n' not in current_content.strip():
            # And if the next block is a table
            if i + 1 < len(blocks) and blocks[i+1][0] == "table":
                # Merge the heading with the table
                table_content = blocks[i+1][1]
                merged_content = current_content + "\n" + table_content
                merged_blocks.append(("table", merged_content))
                i += 2 # Skip the next block since we've merged it
                continue
        # Otherwise, add the block as is
        merged_blocks.append((current_type, current_content))
        i += 1

    # Step 3: Process the final, merged blocks
    for block_type, block_content in merged_blocks:
        if len(re.sub(r'[|\s-]', '', block_content)) < 15: continue

        if block_type == "table":
            docs.append(Document(page_content=block_content, metadata={"source": source_name, "content_type": "table"}))
        else: # block_type == "prose"
            docs.extend(_chunk_text(block_content, source_name, chunker_encoder))
    return docs



# --- Main Public Function ---

def get_documents_from_sources() -> List[Document]:

    print("--- Starting Document Preprocessing and Chunking ---")
    
    # Initialize the chunker's encoder model with CPU-compatible settings
    print(f"Initializing Chunker Encoder: {config.EMBEDDING_MODEL}...")
    
    # CPU optimization: Use appropriate settings based on device
    if config.DEVICE.type == "cpu":
        chunker_encoder = HuggingFaceEncoder(
            name=config.EMBEDDING_MODEL, 
            device="cpu",
            model_kwargs={"torch_dtype": "float32"}
        )
    else:
        chunker_encoder = HuggingFaceEncoder(
            name=config.EMBEDDING_MODEL, 
            device=config.DEVICE
        )

    # Find all source PDFs which are the basis for our processing.
    all_docs = []
    source_pdfs = list(config.PDF_SOURCE_DIR.glob('*.pdf'))
    
    if not source_pdfs:
        print(f"⚠️ No PDF files found in: {config.PDF_SOURCE_DIR}")
        return []

    print(f"Found {len(source_pdfs)} source PDFs to process.")
    
    # CPU optimization: Process smaller batches or limit concurrent operations
    if config.DEVICE.type == "cpu":
        # For CPU, we might want to process fewer documents at once to manage memory
        pass  # The existing logic should work fine, but you could add batch limiting here if needed
    
    for pdf_path in tqdm(source_pdfs, desc="Processing sources"):
        json_path = config.JSON_DEST_DIR / f"{pdf_path.stem}_model.json"
        markdown_path = config.MARKDOWN_DEST_DIR / f"{pdf_path.stem}.md"
        
        source_text = ""
        source_name = pdf_path.name

        # "Markdown-first" strategy:
        if markdown_path.exists():
            tqdm.write(f"  -> Found pre-extracted Markdown for {source_name}. Reading file.")
            raw_html_markdown = markdown_path.read_text(encoding='utf-8')
            tqdm.write("     -> Converting HTML tables to proper Markdown format...")
            source_text = _convert_html_to_markdown(raw_html_markdown)
        else:
            # If no Markdown exists, run the CLI to generate it.
            _run_mineru_cli_and_save(pdf_path)
            # Now, try to read the newly created Markdown file in the same run.
            if markdown_path.exists():
                raw_html_markdown = markdown_path.read_text(encoding='utf-8')
                tqdm.write("     -> Converting HTML tables from newly generated Markdown...")
                source_text = _convert_html_to_markdown(raw_html_markdown)

        if source_text:
            source_text = re.sub(r'!\[.*?\]\(.*?\)', '', source_text) # Clean image links
            docs_from_file = _chunk_document_holistically(source_text, source_name, chunker_encoder)
            all_docs.extend(docs_from_file)
            
    # Add chunk numbers to all documents
    for i, doc in enumerate(all_docs):
        doc.metadata["chunk_number"] = i + 1
            
    print(f"✅ Preprocessing complete. Total chunks created: {len(all_docs)}")
    return all_docs
