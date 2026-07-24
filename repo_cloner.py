import os
import re
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Optional
from git import Repo, GitCommandError
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File extensions to ignore during repository scanning
DEFAULT_EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".avi", ".mov", ".mp3", ".wav",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".pdf", ".docx", ".xlsx",
    ".csv", ".parquet", ".feather", ".db", ".sqlite",
    ".lock", "-lock.json"
}

# Directories to skip during scanning
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".github", ".vscode", "__pycache__", "node_modules",
    "venv", ".venv", "env", "dist", "build", "target", ".idea"
}


class RepoCloneConfig(BaseModel):
    repo_url: str = Field(..., description="HTTPS or SSH repository URL")
    branch: Optional[str] = Field(default=None, description="Specific branch or tag to clone")
    depth: int = Field(default=1, description="Shallow clone depth")
    target_dir: Optional[str] = Field(default=None, description="Target directory for cloning")
    token: Optional[str] = Field(default=None, description="GitHub token for private repos")


class RepoCloner:
    """Handles secure, shallow cloning of Git repositories and file path filtering."""

    @staticmethod
    def validate_url(repo_url: str) -> bool:
        regex = r"^(https?://|git@)(github\.com|gitlab\.com|bitbucket\.org|[\w.-]+)[:/]([\w.-]+)/([\w.-]+?)(\.git)?$"
        return bool(re.match(regex, repo_url.strip()))

    def clone_repository(self, config: RepoCloneConfig) -> Path:
        """Clones a remote repository shallowly into a target or temporary directory."""
        clean_url = config.repo_url.strip()
        
        # Insert token into HTTPS URL if provided for private repos
        if config.token and clean_url.startswith("https://"):
            clean_url = clean_url.replace("https://", f"https://{config.token}@")

        if not config.token and not self.validate_url(config.repo_url):
            raise ValueError(f"Invalid Git URL format: {config.repo_url}")

        clone_path = Path(config.target_dir) if config.target_dir else Path(tempfile.mkdtemp(prefix="repomind_"))
        logger.info(f"Cloning {config.repo_url} into {clone_path}...")

        clone_kwargs = {
            "depth": config.depth,
            "single_branch": True,
            "no_tags": True
        }
        if config.branch and config.branch.strip():
            clone_kwargs["branch"] = config.branch.strip()

        try:
            # If path exists and is non-empty, clean up before cloning
            if clone_path.exists() and any(clone_path.iterdir()):
                shutil.rmtree(clone_path, ignore_errors=True)
                clone_path.mkdir(parents=True, exist_ok=True)

            Repo.clone_from(clean_url, clone_path, **clone_kwargs)
            logger.info("Cloning completed successfully.")
            return clone_path
        except GitCommandError as e:
            logger.error(f"Git clone failed: {e}")
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repo {config.repo_url}: {str(e)}")

    @staticmethod
    def get_valid_code_files(repo_path: Path, allowed_extensions: Optional[List[str]] = None) -> List[Path]:
        """Scans the cloned directory for valid, parseable code files."""
        valid_files = []
        allowed_set = set(ext.lower() for ext in allowed_extensions) if allowed_extensions else None

        for root, dirs, files in os.walk(repo_path):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in DEFAULT_EXCLUDE_DIRS]
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext in DEFAULT_EXCLUDE_EXTENSIONS:
                    continue

                if allowed_set and ext not in allowed_set:
                    continue

                # Skip files larger than 1MB to keep indexing fast
                try:
                    if file_path.stat().st_size > 1_000_000:
                        continue
                except OSError:
                    continue

                valid_files.append(file_path)

        return valid_files
