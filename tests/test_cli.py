from pathlib import Path

from nyaya import cli

ROOT = Path(__file__).resolve().parents[1]


def test_ask_prints_statute_sections(capsys):
    rc = cli.main(["ask", "cheque bounce notice period", "--k", "3",
                   "--canonical-dir", str(ROOT / "data" / "canonical")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Negotiable Instruments Act" in out
    assert "Section" in out
    assert "not legal advice" in out


def test_ask_resolves_an_exact_old_law_citation(capsys):
    cli.main(["ask", "what is Section 420 IPC now", "--k", "2",
              "--canonical-dir", str(ROOT / "data" / "canonical")])
    out = capsys.readouterr().out
    assert "Section 318 of the Bharatiya Nyaya Sanhita" in out
