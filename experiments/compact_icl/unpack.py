"""Extract the BIRD SQLite files and dev labels into DeepEye's data layout."""
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def extract(archive, destination, split, allow_nested=True):
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or any(part.startswith(".") or part == "__MACOSX" for part in path.parts) or entry.is_dir():
            continue
        if path.name == f"{split}_databases.zip" and allow_nested:
            with tempfile.TemporaryFile(dir=destination) as nested_file:
                with archive.open(entry) as source:
                    shutil.copyfileobj(source, nested_file)
                with zipfile.ZipFile(nested_file) as nested:
                    extract(nested, destination, split, False)
            continue
        if path.suffix == ".sqlite":
            target = destination / f"{split}_databases" / path.stem / path.name
        elif split == "dev" and path.name in {"dev.json", "dev_tables.json", "dev_gold.sql"}:
            target = destination / path.name
        elif path.suffix == ".csv" and "database_description" in path.parts:
            index = path.parts.index("database_description")
            if index == 0:
                continue
            target = destination / f"{split}_databases" / path.parts[index - 1] / "database_description" / path.name
        else:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == entry.file_size:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        with archive.open(entry) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
        partial.replace(target)


if __name__ == "__main__":
    root = Path("data/bird")
    for split in ("train", "dev"):
        destination = root / split
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / "downloads" / f"{split}.zip") as archive:
            extract(archive, destination, split)
