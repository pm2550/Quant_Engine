"""SQLite hot backup with local rotation + optional offsite upload.

Local rotation alone protects against schema migration accidents, fsck
corruption, and ransomware (depending on retention).  Offsite copy
protects against full-host loss.  Both are required to call this
"industrial grade".

Layout:
    /data2/quant/db/quant.sqlite                      (live)
    /data2/quant/backups/quant.sqlite.YYYYMMDD.gz     (local, 14-day retention)
    cloudreve://my/quant-backups/quant.sqlite.YYYYMMDD.gz  (offsite, if configured)
"""
from __future__ import annotations
import gzip
import logging
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("quant.db_backup")

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "quant.sqlite"
BACKUP_DIR = ROOT / "backups"
RETENTION_DAYS = 14


def _hot_backup_to(src: Path, dest_uncompressed: Path) -> None:
    """Use sqlite3 .backup API — safe even if the live DB is being written."""
    with sqlite3.connect(src) as src_conn, sqlite3.connect(dest_uncompressed) as dst_conn:
        src_conn.backup(dst_conn)


def _gzip_inplace(path: Path) -> Path:
    gz_path = path.with_suffix(path.suffix + ".gz")
    with open(path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=6) as f_out:
        shutil.copyfileobj(f_in, f_out)
    path.unlink()
    return gz_path


def _prune_old(backup_dir: Path, keep_days: int) -> int:
    cutoff = time.time() - keep_days * 86400
    removed = 0
    for f in backup_dir.glob("quant.sqlite.*.gz"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    return removed


def _verify_gz(path: Path) -> int:
    """Decompress to /dev/null to confirm the gzip is intact. Returns SQLite size."""
    size = 0
    with gzip.open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            size += len(buf)
    return size


def _upload_pmdrive(local_path: Path, filename: str) -> str | None:
    """Best-effort offsite upload to PM Drive. Returns URI or None if not configured."""
    base = os.environ.get("PMDRIVE_BASE_URL")
    email = os.environ.get("PMDRIVE_EMAIL")
    password = os.environ.get("PMDRIVE_PASSWORD")
    root_dir = os.environ.get("PMDRIVE_BACKUP_DIR", "cloudreve://my/quant-backups")
    if not (base and email and password):
        logger.info("pmdrive: credentials not set in env, skipping offsite upload")
        return None

    import httpx
    base = base.rstrip("/")
    with httpx.Client(timeout=300) as c:
        r = c.post(f"{base}/api/v4/session/token",
                   json={"email": email, "password": password})
        r.raise_for_status()
        token = r.json()["data"]["token"]["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        c.post(f"{base}/api/v4/file/create",
               headers={**h, "Content-Type": "application/json"},
               json={"uri": root_dir, "type": "folder"})

        size = local_path.stat().st_size
        uri = f"{root_dir}/{filename}"
        r = c.put(f"{base}/api/v4/file/upload",
                  headers={**h, "Content-Type": "application/json"},
                  json={"uri": uri, "size": size,
                        "last_modified": int(time.time() * 1000),
                        "mime_type": "application/gzip"})
        r.raise_for_status()
        sess = r.json()["data"]
        sid, chunk_size = sess["session_id"], sess["chunk_size"]

        try:
            with open(local_path, "rb") as f:
                idx = 0
                while True:
                    buf = f.read(chunk_size)
                    if not buf and idx > 0:
                        break
                    r = c.post(f"{base}/api/v4/file/upload/{sid}/{idx}",
                               headers={**h, "Content-Type": "application/octet-stream"},
                               content=buf or b"")
                    r.raise_for_status()
                    if not buf:
                        break
                    idx += 1
        except Exception:
            c.request("DELETE", f"{base}/api/v4/file/upload",
                      headers={**h, "Content-Type": "application/json"},
                      json={"id": sid, "uri": uri})
            raise
    return uri


def run_backup() -> dict:
    """Perform the daily backup. Returns a result dict for logging/CLI."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        return {"ok": False, "error": f"db not found: {DB_PATH}"}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    raw_dest = BACKUP_DIR / f"quant.sqlite.{stamp}"

    t0 = time.time()
    _hot_backup_to(DB_PATH, raw_dest)
    gz_path = _gzip_inplace(raw_dest)
    verified_size = _verify_gz(gz_path)

    pruned = _prune_old(BACKUP_DIR, RETENTION_DAYS)

    offsite_uri = None
    offsite_error = None
    try:
        offsite_uri = _upload_pmdrive(gz_path, gz_path.name)
    except Exception as e:
        offsite_error = repr(e)
        logger.warning("offsite upload failed: %s", e)

    return {
        "ok": True,
        "local_path": str(gz_path),
        "compressed_bytes": gz_path.stat().st_size,
        "uncompressed_bytes": verified_size,
        "elapsed_sec": round(time.time() - t0, 2),
        "pruned_old": pruned,
        "offsite_uri": offsite_uri,
        "offsite_error": offsite_error,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = run_backup()
    if not res.get("ok"):
        print(f"FAIL: {res}", file=sys.stderr)
        return 2
    print(
        f"OK: {res['local_path']} "
        f"({res['compressed_bytes']/1e6:.1f} MB gz, "
        f"verified {res['uncompressed_bytes']/1e6:.1f} MB), "
        f"elapsed {res['elapsed_sec']}s, pruned {res['pruned_old']} old."
    )
    if res["offsite_uri"]:
        print(f"OFFSITE: {res['offsite_uri']}")
    elif res["offsite_error"]:
        print(f"OFFSITE FAIL: {res['offsite_error']}", file=sys.stderr)
        return 1
    else:
        print("OFFSITE: skipped (PMDRIVE_* env not set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
