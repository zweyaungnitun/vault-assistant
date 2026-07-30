import pytest
from fastapi.testclient import TestClient

from vault_assistant.api import create_app
from vault_assistant.config import Config
from vault_assistant.ingest import ingest_paths

from conftest import FakeClient


@pytest.fixture
def app(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = Config(data_dir=data_dir)
    application = create_app(cfg)
    application.state.client = FakeClient()
    return application


@pytest.fixture
def api(app):
    return TestClient(app)


def _write(path, text):
    path.write_text(text)


def test_folders_crud(tmp_path, api):
    folder = tmp_path / "docs"
    folder.mkdir()

    r = api.post("/api/folders", json={"path": str(folder), "access_level": "readonly"})
    assert r.status_code == 200
    created = r.json()
    assert created["access_level"] == "readonly"

    r = api.get("/api/folders")
    assert r.status_code == 200
    listed = r.json()
    assert len(listed) == 1
    assert listed[0]["document_count"] == 0

    r = api.put(f"/api/folders/{created['id']}", json={"access_level": "edit"})
    assert r.status_code == 200
    assert r.json()["access_level"] == "edit"

    r = api.delete(f"/api/folders/{created['id']}")
    assert r.status_code == 200
    assert api.get("/api/folders").json() == []


def test_folder_create_rejects_non_directory(tmp_path, api):
    r = api.post("/api/folders", json={"path": str(tmp_path / "nope"), "access_level": "edit"})
    assert r.status_code == 400


def test_folder_update_rejects_unknown_level(tmp_path, api):
    folder = tmp_path / "docs"
    folder.mkdir()
    created = api.post("/api/folders", json={"path": str(folder), "access_level": "edit"}).json()

    r = api.put(f"/api/folders/{created['id']}", json={"access_level": "superuser"})
    assert r.status_code == 400


def test_folder_update_missing_id_is_404(api):
    r = api.put("/api/folders/999", json={"access_level": "edit"})
    assert r.status_code == 404


def test_document_delete_requires_edit_access(tmp_path, app, api):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "a.txt", "Some readonly content. " * 20)
    ingest_paths(app.state.conn, app.state.client, [folder])
    doc_id = api.get("/api/documents").json()[0]["id"]

    api.post("/api/folders", json={"path": str(folder), "access_level": "readonly"})
    assert api.get("/api/documents").json()[0]["access_level"] == "readonly"

    r = api.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 403
    assert len(api.get("/api/documents").json()) == 1

    folder_row = api.get("/api/folders").json()[0]
    api.put(f"/api/folders/{folder_row['id']}", json={"access_level": "edit"})

    r = api.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert api.get("/api/documents").json() == []


def test_force_reingest_blocked_for_readonly_folder(tmp_path, api):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "a.txt", "Some content. " * 20)
    api.post("/api/folders", json={"path": str(folder), "access_level": "readonly"})

    r = api.post("/api/ingest", json={"paths": [str(folder)], "force": True})
    assert r.status_code == 403

    r = api.post("/api/ingest", json={"paths": [str(folder)], "force": False})
    assert r.status_code == 200
    assert r.json()["added"] == 1


def test_no_access_folder_excludes_documents_via_api(tmp_path, app, api):
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(folder / "a.txt", "Some content. " * 20)
    ingest_paths(app.state.conn, app.state.client, [folder])
    assert len(api.get("/api/documents").json()) == 1

    created = api.post("/api/folders", json={"path": str(folder), "access_level": "edit"}).json()
    api.put(f"/api/folders/{created['id']}", json={"access_level": "no_access"})

    assert api.get("/api/documents").json() == []


def test_ingest_reports_blocked_count(tmp_path, api):
    folder = tmp_path / "docs"
    (folder / "private").mkdir(parents=True)
    _write(folder / "visible.txt", "visible content here " * 10)
    _write(folder / "private" / "secret.txt", "secret content here " * 10)
    api.post("/api/folders", json={"path": str(folder / "private"), "access_level": "no_access"})

    r = api.post("/api/ingest", json={"paths": [str(folder)]})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 1
    assert body["blocked"] == 1
