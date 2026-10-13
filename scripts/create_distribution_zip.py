"""
Distribution Packaging Script for FloodAI Engine
Creates a clean, reproducible, publication-ready ZIP archive of the project
excluding git history, cache directories, and temporary bytecode.
"""

import os
import sys
import zipfile
from pathlib import Path

# Directories to exclude
EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".idea",
    ".vscode",
    ".coverage",
}

# File extensions or specific filenames to exclude
EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
}


def create_zip(source_dir: Path, output_zip: Path):
    print(f"Packaging FloodAI Engine from: {source_dir}")
    print(f"Target archive: {output_zip}")
    
    file_count = 0
    total_uncompressed_bytes = 0

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # In-place modify dirs to avoid recursing into excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".git")]
            
            rel_root = Path(root).relative_to(source_dir)
            
            for file in files:
                if file in EXCLUDE_FILES:
                    continue
                if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                    continue
                
                # Avoid including the zip file if written within source
                file_path = Path(root) / file
                if file_path.resolve() == output_zip.resolve():
                    continue
                
                archive_name = Path("flood_prediction_resilience") / rel_root / file
                zipf.write(file_path, archive_name.as_posix())
                file_count += 1
                total_uncompressed_bytes += file_path.stat().st_size

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)
    print("\n" + "=" * 60)
    print("PACKAGE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Files Packaged: {file_count}")
    print(f"Uncompressed Size:    {raw_size_mb:.2f} MB")
    print(f"Compressed ZIP Size:  {zip_size_mb:.2f} MB")
    print(f"Archive Location:     {output_zip.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    default_output = base_dir.parent / "flood_prediction_resilience.zip"
    
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_output
    create_zip(base_dir, out_path)
