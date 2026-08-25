from pathlib import Path

from skai.ingest.loaders import load_markdown, load_pdf, load_sources, load_web

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_markdown():
    docs = load_markdown(FIXTURES / "sample.md")
    assert len(docs) == 1
    d = docs[0]
    assert d.metadata["source_type"] == "md"
    assert d.metadata["title"] == "Orca Whales"
    assert "killer whales" in d.text


def test_load_web_extracts_article_and_drops_chrome():
    html = (FIXTURES / "article.html").read_text()
    docs = load_web("https://example.com/orcas", _html=html)
    assert len(docs) == 1
    d = docs[0]
    assert d.metadata["source_type"] == "web"
    assert d.metadata["uri"] == "https://example.com/orcas"
    assert d.metadata["title"] == "Pod Communication"
    assert "dialect" in d.text
    # navigation / footer boilerplate must be stripped by trafilatura
    assert "Home | About" not in d.text
    assert "Copyright" not in d.text


def test_load_pdf(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Orcas hunt in coordinated pods.")
    out = tmp_path / "orca.pdf"
    pdf.output(str(out))

    docs = load_pdf(out)
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "pdf"
    assert docs[0].metadata["source_id"] == "orca.pdf#p1"
    assert "Orcas hunt" in docs[0].text


def test_load_sources_skips_failures(tmp_path, caplog):
    (tmp_path / "good.md").write_text("# Good\nreal content here")
    (tmp_path / "broken.pdf").write_text("not a real pdf")  # will fail to parse
    docs = load_sources(tmp_path, urls=[])
    ids = [d.metadata["source_id"] for d in docs]
    assert any("good.md" in i for i in ids)
    assert all("broken.pdf" not in i for i in ids)  # skipped, not crashed
