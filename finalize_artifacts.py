"""Check executed notebook, preserve its image alternatives, and package outputs.

Raw face datasets, downloaded archives, the original PDF, caches, and kernel
connection files are excluded. Use download_data.py to restore face data.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent


def main():
    notebook = json.loads((ROOT / "reproduction.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code_cells for o in c.get("outputs", []) if o["output_type"] == "error"]
    if errors or any(c["execution_count"] is None for c in code_cells):
        raise ValueError("Notebook must have executed all code cells without errors")
    alternatives = [o["metadata"]["image/png"]["alt"] for c in code_cells
                    for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    html_path = ROOT / "reproduction.html"
    html = html_path.read_text(encoding="utf-8")
    # This installed nbconvert version ignores Image's nested MIME alt metadata.
    placeholder = 'alt="No description has been provided for this image"'
    if placeholder in html:
        if html.count(placeholder) != len(alternatives):
            raise ValueError("HTML image count differs from executed notebook")
        for alternative in alternatives:
            html = html.replace(placeholder, f'alt="{escape(alternative, quote=True)}"', 1)
        html_path.write_text(html, encoding="utf-8")

    audit = json.loads((ROOT / "outputs/validation_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "pass":
        raise ValueError("Independent experiment record validation did not pass")
    manifest_path = ROOT / "outputs/reproduction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = sorted([*ROOT.glob("*.py"), *ROOT.glob("enlbp/*.py"),
                      *ROOT.glob("tests/test_*.py"), ROOT/"requirements.txt", ROOT/"pytest.ini"])
    manifest["code_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources}
    manifest["notebook_validation"] = {"executed_code_cells": len(code_cells), "errors": len(errors)}
    manifest["independent_result_audit_status"] = audit["status"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    paths = [*sources, ROOT/"README.md", ROOT/"实验结果.md", ROOT/"reproduction.ipynb",
             ROOT/"reproduction.html", ROOT/".gitignore"]
    paths += sorted(p for p in (ROOT/"paper").iterdir()
                    if p.is_file() and p.name != "paper_text.txt" and p.suffix.lower() != ".pdf")
    paths += sorted(p for p in (ROOT/"data").glob("*_source.json"))
    paths += sorted(p for p in (ROOT/"outputs").rglob("*") if p.is_file())
    paths = sorted(set(paths))
    archive = ROOT / "reproduction_bundle.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for path in paths:
            z.write(path, arcname=path.relative_to(ROOT).as_posix())
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Bundle ZIP integrity check failed")
    print(f"Notebook: {len(code_cells)} executed cells, 0 errors; audit: pass.")
    print(f"Bundle: {len(paths)} files, {archive.stat().st_size:,} bytes.")


if __name__ == "__main__":
    main()
