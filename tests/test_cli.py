from typer.testing import CliRunner

from skai import cli
from skai.config import get_settings
from skai.ingest.store import Store

runner = CliRunner()


def test_ingest_populates_store(tmp_path, ef, monkeypatch):
    # point config at tmp dirs, inject offline embeddings
    monkeypatch.setenv("SKAI_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("SKAI_COLLECTION", "cli_kb")
    get_settings.cache_clear()

    store = Store(str(tmp_path / "chroma"), "cli_kb", embedding_function=ef)
    monkeypatch.setattr(cli, "_open_store", lambda settings: store)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("# Orcas\nOrcas hunt seals in coordinated pods.")

    result = runner.invoke(cli.app, ["ingest", "--path", str(docs), "--urls", "nope.txt"])
    assert result.exit_code == 0, result.output
    assert "ingested" in result.output
    assert store.count() > 0


def test_ask_help_lists_source_option():
    result = runner.invoke(cli.app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output


def test_top_level_help_lists_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "ask", "chat", "mcp", "eval"):
        assert cmd in result.output
