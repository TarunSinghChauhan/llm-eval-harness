from src.monitoring.regression import select_alert_emoji, select_severity_icon


def test_select_alert_emoji_critical():
    assert select_alert_emoji(True) == "\U0001F6A8"


def test_select_alert_emoji_non_critical():
    assert select_alert_emoji(False) == "\u26A0\uFE0F"


def test_select_severity_icon_critical():
    assert select_severity_icon("critical") == "\U0001F534"


def test_select_severity_icon_warning():
    assert select_severity_icon("warning") == "\U0001F7E1"


def test_select_severity_icon_unknown_defaults_to_yellow():
    assert select_severity_icon("something_else") == "\U0001F7E1"
