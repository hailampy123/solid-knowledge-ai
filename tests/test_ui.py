"""Guard the Gradio wiring: build_ui must construct (any bad component param
raises here). No server is started."""


def test_build_ui_constructs():
    from skai.ui import CSS, THEME, build_ui

    demo = build_ui()
    assert demo.__class__.__name__ == "Blocks"
    assert THEME is not None and CSS.strip()
