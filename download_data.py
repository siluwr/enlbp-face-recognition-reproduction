"""Download source-linked ORL or LFW-a images, validate ZIP, and extract safely.

No downloaded content is executed. Google Drive's large-file confirmation page
is parsed only for its download form; authentication pages are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

SOURCES = {
    "orl": {
        "url": "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip",
        "description_url": "https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html",
        "credit": "AT&T Laboratories Cambridge",
        "expected_images": 400,
        "expected_subjects": 40,
        "expected_sha256": "509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec",
    },
    "lfwa": {
        "url": "https://drive.google.com/uc?export=download&id=1p1wjaqpTh_5RHfJu4vUh8JJCdKwYMHCp",
        "description_url": "https://talhassner.github.io/home/projects/lfwa/",
        "credit": "Lior Wolf, Tal Hassner, Yaniv Taigman; original LFW creators",
        "expected_images": 13233,
        "expected_subjects": 5749,
    },
}


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form" and a.get("id") == "download-form":
            self.active = True
            self.action = a.get("action")
        if self.active and tag == "input" and a.get("type") == "hidden":
            self.fields[a["name"]] = a.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def fetch(url: str, output: Path) -> None:
    request = Request(url, headers={"User-Agent": "EnLBP-reproduction/1.0"})
    with urlopen(request, timeout=180) as response, output.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def extract_images(archive: Path, root: Path) -> int:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    with ZipFile(archive) as z:
        for item in z.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP path: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symbolic links are unsupported")
            if item.is_dir():
                continue
            if target.suffix.lower() not in {".pgm", ".jpg", ".jpeg", ".png", ".txt"}:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            if target.suffix.lower() != ".txt":
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--archive", type=Path, help="Use an already downloaded ZIP")
    args = parser.parse_args()
    spec = SOURCES[args.dataset]
    archive = args.archive or args.data_root / "downloads" / f"{args.dataset}.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".part")
        print(f"Downloading {args.dataset} from its source-linked URL...", flush=True)
        fetch(spec["url"], temporary)
        if not is_zipfile(temporary) and args.dataset == "lfwa":
            if temporary.stat().st_size > 1024 * 1024:
                raise ValueError("Unexpected non-ZIP response")
            form = DownloadForm()
            form.feed(temporary.read_text(encoding="utf-8", errors="replace"))
            if form.action != "https://drive.usercontent.google.com/download":
                raise ValueError("Download unavailable; obtain LFW-a from its author page")
            fetch(form.action + "?" + urlencode(form.fields), temporary)
        if not is_zipfile(temporary):
            raise ValueError("Source returned no ZIP archive")
        temporary.replace(archive)
    if not is_zipfile(archive):
        raise ValueError("Input is not a valid ZIP archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if spec.get("expected_sha256") and digest != spec["expected_sha256"]:
        raise ValueError("ORL archive differs from the verified source archive")
    root = args.data_root / args.dataset
    count = extract_images(archive, root)
    if count != spec["expected_images"]:
        raise ValueError(f"Expected {spec['expected_images']} images, extracted {count}")
    manifest = {**spec, "archive_sha256": digest, "extracted_images": count,
                "download_date": datetime.now().astimezone().isoformat(),
                "archive_name": archive.name,
                "data_directory": args.dataset}
    manifest_path = args.data_root / f"{args.dataset}_source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Extracted {count} images to {root}. SHA-256: {digest}")


if __name__ == "__main__":
    main()
