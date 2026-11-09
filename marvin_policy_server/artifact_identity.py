"""Identity of the exact bytes given to a loader, not a later disk snapshot."""

import hashlib
from pathlib import Path


def artifact_identity(path, content: bytes) -> dict:
    return {
        'path': str(Path(path).absolute()),
        'sha256': hashlib.sha256(content).hexdigest(),
        'size': len(content),
    }
