"""Versioned prompt configuration loader for TeleRAG."""

import os
from dataclasses import dataclass

import yaml

REQUIRED_PROMPT_KEYS = [
    "version",
    "system_prompt",
    "user_template",
    "refusal_message",
    "insufficient_context_marker",
]


@dataclass
class PromptConfig:
    """Container for loaded prompt templates and markers."""

    version: str
    system_prompt: str
    user_template: str
    refusal_message: str
    insufficient_context_marker: str

    def format_user_prompt(self, context: str, question: str) -> str:
        """Format user prompt using user_template."""
        return self.user_template.format(context=context, question=question)


def get_default_prompts_dir() -> str:
    """Find the prompts directory relative to project root or cwd."""
    cwd_prompts = os.path.join(os.getcwd(), "prompts")
    if os.path.isdir(cwd_prompts):
        return cwd_prompts
    pkg_prompts = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts"))
    if os.path.isdir(pkg_prompts):
        return pkg_prompts
    return "prompts"


def load_prompt_config(
    prompt_version: str | None = None, prompts_dir: str | None = None
) -> PromptConfig:
    """Load and validate prompt configuration from YAML.

    Args:
        prompt_version: Version identifier (e.g. 'v1'). Defaults to env PROMPT_VERSION or 'v1'.
        prompts_dir: Directory containing prompt YAML files. Defaults to project 'prompts/'.

    Returns:
        PromptConfig object with validated fields.

    Raises:
        FileNotFoundError: If the specified prompt YAML does not exist.
        ValueError: If required keys are missing or content is empty.
    """
    if isinstance(prompt_version, str) and prompt_version.strip():
        version = prompt_version.strip()
    else:
        version = os.getenv("PROMPT_VERSION", "v1").strip() or "v1"
    base_dir = prompts_dir or get_default_prompts_dir()
    prompt_file = os.path.join(base_dir, f"{version}.yaml")

    if not os.path.isfile(prompt_file):
        raise FileNotFoundError(f"Prompt configuration file not found: {prompt_file}")

    try:
        with open(prompt_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML file {prompt_file}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Prompt configuration in {prompt_file} must be a dictionary/mapping, got {type(data).__name__}"
        )

    missing_keys = [k for k in REQUIRED_PROMPT_KEYS if k not in data or data[k] is None]
    if missing_keys:
        raise ValueError(
            f"Prompt configuration in {prompt_file} missing required keys: {missing_keys}"
        )

    return PromptConfig(
        version=str(data["version"]),
        system_prompt=str(data["system_prompt"]),
        user_template=str(data["user_template"]),
        refusal_message=str(data["refusal_message"]),
        insufficient_context_marker=str(data["insufficient_context_marker"]),
    )
