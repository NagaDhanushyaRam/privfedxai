import zipfile
from pathlib import Path

ZIP = "data/raw/ptb-xl-plus-1.0.1.zip"
OUT = Path("data/raw/ptbxl_plus")      # inside data/raw, so already git-ignored
PREFIX = "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/"


def wanted(rel):
    if rel.startswith("median_beats/") or rel.startswith("features/old/"):
        return False
    if rel.startswith("fiducial_points/ecgdeli/"):
        return rel.endswith("_points_global.atr")
    return True                          # RECORDS, LICENSE, SHA256SUMS, features/*.csv, labels/*


if __name__ == "__main__":
    z = zipfile.ZipFile(ZIP)
    names = [n for n in z.namelist() if not n.endswith("/") and n.startswith(PREFIX)]
    todo = [n for n in names if wanted(n[len(PREFIX):])]
    print("in zip:", len(names), "to extract:", len(todo))
    for i, n in enumerate(todo):
        t = OUT / n[len(PREFIX):]        # prefix stripped: shorter Windows paths
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(z.read(n))
        if i % 5000 == 0:
            print(i, flush=True)
    print("done")