from vault_assistant.pii import locate_all, model_scan, regex_scan, scan


def _kinds_covering(spans, text, needle):
    idx = text.find(needle)
    assert idx != -1
    return [s.kind for s in spans if s.start <= idx and s.end >= idx + len(needle)]


def test_regex_email_phone_ssn():
    text = (
        "Contact Jane at jane.doe@example.com or 555-123-4567. "
        "Her SSN is 123-45-6789."
    )
    spans = regex_scan(text)
    assert _kinds_covering(spans, text, "jane.doe@example.com") == ["email"]
    assert _kinds_covering(spans, text, "555-123-4567")
    assert _kinds_covering(spans, text, "123-45-6789")


def test_regex_card_and_account():
    text = "Card 4111 1111 1111 1111, account 12345678, IBAN DE89370400440532013000."
    spans = regex_scan(text)
    assert _kinds_covering(spans, text, "4111 1111 1111 1111")
    assert _kinds_covering(spans, text, "12345678")
    assert _kinds_covering(spans, text, "DE89370400440532013000")


def test_spans_have_exact_offsets():
    text = "email me: a@b.co thanks"
    spans = regex_scan(text)
    assert len(spans) == 1
    s = spans[0]
    assert text[s.start:s.end] == s.text == "a@b.co"


def test_no_pii_no_spans():
    assert regex_scan("The quarterly report is due soon.") == []


def test_locate_all_finds_every_occurrence():
    text = "Bob met Bob. Bob left."
    assert locate_all(text, "Bob") == [(0, 3), (8, 11), (13, 16)]
    assert locate_all(text, "") == []
    assert locate_all(text, "Alice") == []


def test_model_scan_locates_quoted_strings(client):
    text = "Meeting with John Smith at 12 Elm Street tomorrow."
    client.json_response = {
        "items": [
            {"text": "John Smith", "kind": "name"},
            {"text": "12 Elm Street", "kind": "address"},
            {"text": "not actually present", "kind": "name"},
        ]
    }
    spans = model_scan(client, text)
    found = {(s.text, s.kind) for s in spans}
    assert ("John Smith", "name") in found
    assert ("12 Elm Street", "address") in found
    # hallucinated strings that don't appear in the text are dropped
    assert all("not actually" not in s.text for s in spans)


def test_scan_merges_regex_and_model(client):
    text = "John Smith, john@smith.io"
    client.json_response = {"items": [{"text": "John Smith", "kind": "name"}]}
    spans = scan(text, client=client, use_model=True)
    kinds = {s.kind for s in spans}
    assert {"name", "email"} <= kinds
    # sorted, non-contained
    assert spans == sorted(spans, key=lambda s: s.start)


def test_scan_regex_only(client):
    text = "John Smith, john@smith.io"
    spans = scan(text, client=client, use_model=False)
    assert {s.kind for s in spans} == {"email"}
