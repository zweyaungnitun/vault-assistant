from vault_assistant.actions import extract_actions


def test_extracts_items_and_normalizes(client):
    client.json_response = {
        "items": [
            {
                "task": "Send the budget report",
                "owner": "Maria",
                "due_date": "Friday",
                "source_snippet": "Maria will send the budget report by Friday",
            },
            {"task": "  ", "owner": None, "due_date": None, "source_snippet": ""},
            {"task": "Book the venue", "owner": "", "due_date": "", "source_snippet": "book the venue"},
        ]
    }
    items = extract_actions(client, "some meeting notes")
    assert len(items) == 2  # blank task dropped
    assert items[0].task == "Send the budget report"
    assert items[0].owner == "Maria"
    assert items[1].owner is None  # empty string becomes None, not invented
    assert items[1].due_date is None


def test_empty_result(client):
    client.json_response = {"items": []}
    assert extract_actions(client, "nothing actionable here") == []
