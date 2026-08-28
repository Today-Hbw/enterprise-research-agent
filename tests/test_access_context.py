from app.main import access_context_from_headers, settings


def test_access_headers_are_ignored_until_trusted_gateway_mode_is_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "knowledge_trust_access_headers", False)

    untrusted = access_context_from_headers("forged-tenant", "mallory")

    assert untrusted.tenant_id == settings.knowledge_default_tenant
    assert untrusted.principal_ids == {settings.knowledge_default_principal}

    monkeypatch.setattr(settings, "knowledge_trust_access_headers", True)

    trusted = access_context_from_headers("tenant-a", "alice,analyst")

    assert trusted.tenant_id == "tenant-a"
    assert trusted.principal_ids == {"alice", "analyst"}
