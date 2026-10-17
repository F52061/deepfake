import argparse
import json
from pathlib import Path

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
dependency_root = Path("C:/Users/20236/.cache/codex-runtimes/codex-primary-runtime/dependencies")
poppler_root = dependency_root / "native/poppler/Library/bin"
if not (poppler_root / "pdftoppm.exe").is_file():
    poppler_root = next((dependency_root / "native/poppler").rglob("pdftoppm.exe")).parent
pdf_path = args.directory / "render.pdf"
metadata = pdfinfo_from_path(str(pdf_path), poppler_path=str(poppler_root))
images = convert_from_path(str(pdf_path), dpi=150, fmt="png", output_folder=str(args.directory),
                           output_file="page", paths_only=True, poppler_path=str(poppler_root))
page_paths = []
for index, source in enumerate(images, start=1):
    destination = args.directory / f"page-{index}.png"
    Path(source).rename(destination)
    page_paths.append(destination)
for group_start in range(0, len(page_paths), 6):
    selected = page_paths[group_start:group_start + 6]
    contact = Image.new("RGB", (900, 2 * 450), "#dddddd")
    draw = ImageDraw.Draw(contact)
    for index, page_path in enumerate(selected):
        with Image.open(page_path) as page:
            page.thumbnail((285, 410))
            left = (index % 3) * 300 + 7
            top = (index // 3) * 450 + 27
            contact.paste(page, (left, top))
            draw.text((left, top - 20), f"PAGE {group_start + index + 1}", fill="black")
    contact.save(args.directory / f"contact-{group_start // 6 + 1}.png")
(args.directory / "render_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Rendered {len(images)} page PNGs at 150 dpi")
