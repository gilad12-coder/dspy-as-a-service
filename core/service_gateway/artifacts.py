import base64
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from ..exceptions import ServiceError
from ..models import ProgramArtifact

logger = logging.getLogger(__name__)


def persist_program(
    program: Any,
    artifact_id: Optional[str],
    artifacts_root: Path = None,
) -> Optional[ProgramArtifact]:
    """Save the compiled DSPy program to temp, encode to base64, then cleanup.

    All artifact data is returned in the ProgramArtifact for Redis storage.
    No files are left on disk.

    Args:
        program: Compiled DSPy module produced by an optimizer.
        artifact_id: Optional identifier for the artifact.
        artifacts_root: Ignored - kept for backward compatibility.

    Returns:
        Optional[ProgramArtifact]: Serialized metadata bundle with base64 program.

    Raises:
        ServiceError: When saving or reading the artifact fails.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix=f"dspy_artifact_{artifact_id or uuid4()}_")
        destination = Path(temp_dir)

        try:
            program.save(str(destination), save_program=True)
        except Exception as exc:
            logger.exception("Failed to save DSPy program artifact")
            raise ServiceError(f"Failed to save program artifact: {exc}") from exc

        try:
            metadata_path = destination / "metadata.json"
            program_path = destination / "program.pkl"
            metadata = json.loads(metadata_path.read_text())
            program_bytes = program_path.read_bytes()
            program_b64 = base64.b64encode(program_bytes).decode("ascii")
        except Exception as exc:
            logger.exception("Failed to read DSPy artifact contents")
            raise ServiceError(f"Failed to load program artifact contents: {exc}") from exc

        return ProgramArtifact(
            path=None,  # No disk path - everything in Redis
            metadata=metadata,
            program_pickle_base64=program_b64,
        )

    finally:
        # Always cleanup temp directory
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug("Cleaned up temp artifact directory: %s", temp_dir)
