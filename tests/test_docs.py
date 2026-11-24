"""Tests that keep the documentation and the deployment files honest.

They cannot prove the docs are good, but they catch the mistakes that happened before:
settings missing from the docs, links to files that do not exist, and statements about the
code that are not true.
"""

import dataclasses
import os
import re
import unittest

import yaml
from dotenv import dotenv_values

from src.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(relative_path):
    with open(os.path.join(ROOT, relative_path), encoding="utf-8") as f:
        return f.read()


def markdown_files():
    names = ["README.md"]
    names += [f"docs/{name}" for name in sorted(os.listdir(os.path.join(ROOT, "docs")))]
    return [name for name in names if name.endswith(".md")]


def setting_names():
    """Environment variable names, one per Config field (TOP_K_RESULTS for top_k_results)."""
    return {field.name.upper() for field in dataclasses.fields(Config)}


class TestSettingsAreDocumented(unittest.TestCase):
    def test_every_setting_is_in_the_readme_table_and_nothing_extra_is(self):
        rows = re.findall(r"^\| `([A-Z0-9_]+)` \|", read("README.md"), flags=re.MULTILINE)

        self.assertEqual(len(rows), len(set(rows)), "a setting appears twice in the README table")
        self.assertEqual(set(rows), setting_names())

    def test_env_example_lists_every_setting_and_nothing_extra(self):
        listed = set(dotenv_values(os.path.join(ROOT, ".env.example")))

        self.assertEqual(listed, setting_names())

    def test_env_example_has_a_comment_for_the_access_and_admin_settings(self):
        text = read(".env.example")

        for name in ("ALLOWED_USER_IDS", "ADMIN_USER_IDS"):
            self.assertRegex(text, rf"#.*{name}", f"{name} has no explanation in .env.example")


class TestLinks(unittest.TestCase):
    def test_relative_links_point_to_files_that_exist(self):
        broken = []
        for name in markdown_files():
            base = os.path.dirname(os.path.join(ROOT, name))
            for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", read(name)):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path = os.path.normpath(os.path.join(base, target.split("#")[0]))
                if not os.path.exists(path):
                    broken.append(f"{name} -> {target}")

        self.assertEqual(broken, [])

    def test_readme_links_to_the_interview_notes(self):
        self.assertIn("docs/INTERVIEW_NOTES.md", read("README.md"))
        self.assertTrue(os.path.exists(os.path.join(ROOT, "docs", "INTERVIEW_NOTES.md")))


class TestNoKnownFalseClaims(unittest.TestCase):
    """Phrases that were once written about the code but are not true."""

    FALSE_CLAIMS = {
        "DNS pinning": "the code does not pin DNS (see the SSRF limitation)",
        "NLI CrossEncoder)": "the grounding check in the bot uses embedding similarity, not NLI",
        "NLI entailment threshold": "GROUNDING_THRESHOLD is a cosine similarity threshold",
        "RETRIEVAL_THRESHOLD": "the setting is called SIMILARITY_THRESHOLD",
        "Raw Documents": "uploaded files are not stored, only their chunks and embeddings",
        "PyPDF2": "the project uses pypdf",
        "~18 s": "eval/results/llm_bench.json says the median is 13.8 s",
        "20 to 40 s": "no result file supports this; see the 16-question manual test",
        "20–40 s": "no result file supports this; see the 16-question manual test",
        "209 unit tests": "the test count changed",
        "199 unit tests": "the test count changed",
    }

    def test_docs_do_not_contain_claims_that_are_known_to_be_false(self):
        found = []
        for name in markdown_files():
            text = read(name)
            for phrase, reason in self.FALSE_CLAIMS.items():
                if phrase in text:
                    found.append(f"{name}: '{phrase}' ({reason})")

        self.assertEqual(found, [])

    def test_the_readme_says_how_grounding_really_works(self):
        text = read("README.md").lower()

        self.assertIn("cosine similarity", text)
        self.assertIn("not a proof", text)


class TestDockerFiles(unittest.TestCase):
    """Docker cannot be run on every machine, so check the properties that matter."""

    def test_dockerfile_runs_as_a_normal_user(self):
        users = re.findall(r"^USER\s+(\S+)", read("Dockerfile"), flags=re.MULTILINE)

        self.assertTrue(users, "the Dockerfile never switches user")
        self.assertNotIn(users[-1], ("root", "0"))

    def test_dockerfile_installs_cpu_only_pytorch_and_never_copies_the_env_file(self):
        dockerfile = read("Dockerfile")

        self.assertIn("download.pytorch.org/whl/cpu", dockerfile)
        self.assertNotRegex(dockerfile, r"(?im)^COPY\s+.*\.env(\s|$)")
        # Comments may mention CUDA (to explain why it is avoided); only instructions count.
        instruction_lines = [
            line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")
        ]
        self.assertNotRegex(" ".join(instruction_lines).lower(), r"cuda|nvidia")

    def test_dockerignore_keeps_secrets_and_data_out_of_the_image(self):
        ignored = read(".dockerignore").split()

        for entry in (".env", "data", "logs", ".git"):
            self.assertIn(entry, ignored)

    def test_constraints_file_pins_exact_versions(self):
        lines = [
            line.strip()
            for line in read("requirements/docker-constraints.txt").splitlines()
            if line.strip() and not line.startswith("#")
        ]

        self.assertTrue(lines)
        for line in lines:
            self.assertRegex(line, r"^[A-Za-z0-9_.\-]+==\S+$")

    def test_compose_runs_the_bot_with_ollama_and_persistent_volumes(self):
        compose = yaml.safe_load(read("docker-compose.yml"))
        services = compose["services"]

        self.assertEqual(set(services), {"ollama", "ollama-pull", "bot"})
        self.assertEqual(services["bot"]["depends_on"]["ollama"]["condition"], "service_healthy")
        self.assertEqual(
            services["bot"]["depends_on"]["ollama-pull"]["condition"],
            "service_completed_successfully",
        )
        mounted = {volume.split(":")[0] for volume in services["bot"]["volumes"]}
        self.assertEqual(mounted, {"bot_data", "bot_logs"})
        self.assertIn("ollama_models", compose["volumes"])
        for name, service in services.items():
            self.assertNotIn("ports", service, f"{name} publishes a port to the host")

    def test_compose_only_sets_real_settings_and_stays_cpu_only(self):
        text = read("docker-compose.yml")
        compose = yaml.safe_load(text)

        for name in compose["services"]["bot"]["environment"]:
            self.assertIn(name, setting_names())
        self.assertNotRegex(text.lower(), r"nvidia|gpus:|devices:")


class TestCiWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(read(".github/workflows/ci.yml"))

    def test_runs_on_every_push(self):
        triggers = self.workflow.get("on") or self.workflow.get(True)

        self.assertIn("push", triggers)
        push = triggers["push"] or {}
        self.assertNotIn("branches", push, "push is limited to some branches")

    def test_runs_lint_format_check_and_tests(self):
        commands = " ".join(
            step.get("run", "") for job in self.workflow["jobs"].values() for step in job["steps"]
        )

        self.assertIn("ruff check", commands)
        self.assertIn("ruff format --check", commands)
        self.assertIn("unittest discover tests", commands)


if __name__ == "__main__":
    unittest.main()
