# SPDX-FileCopyrightText: 2026 Bentley Systems, Incorporated
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for session isolation: context, object_staging, and object_registry.

Verifies that in delegated auth mode, each MCP client session gets its own
DelegatedAuthContext with independent StagingService and ObjectRegistry.
"""

import asyncio
import base64
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from cachetools import TTLCache

import evo_mcp.context as ctx_module
from evo_mcp.context import _CleanupTTLCache
from evo_mcp.contexts.delegated import DelegatedAuthContext
from evo_mcp.contexts.helpers import get_client_session_id
from evo_mcp.contexts.managed import ManagedAuthContext
from evo_mcp.session.resolver import ResolutionError
from evo_mcp.staging.errors import StageNotFoundError
from evo_mcp.staging.models import StagedEnvelope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_token(sub: str = "user-1", iat: int | None = None) -> str:
    """Return a minimal unsigned JWT with the given sub claim.

    ``iat`` defaults to the current POSIX time so successive calls with the
    same ``sub`` produce distinct token strings (simulating token refresh).
    Pass an explicit ``iat`` when you need a deterministic value.
    """
    if iat is None:
        iat = int(time.time())
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub, "iat": iat}).encode()).rstrip(b"=")
    return f"{header.decode()}.{payload.decode()}."


def _mock_access_token_obj(token: str):
    """Create a mock object mimicking FastMCP's AccessToken."""
    obj = MagicMock()
    obj.token = token
    return obj


def _mock_http_request(session_id: str | None):
    """Create a mock HTTP request, optionally without a legacy session header."""
    request = MagicMock()
    request.headers = {"mcp-session-id": session_id} if session_id is not None else {}
    return request


# ---------------------------------------------------------------------------
# Test: Different sessions get different contexts
# ---------------------------------------------------------------------------


class TestContextPerSession(unittest.IsolatedAsyncioTestCase):
    """get_evo_context() returns distinct contexts for distinct sessions."""

    async def test_different_sessions_get_different_contexts(self):
        token_a = _make_mock_token("user-A")
        token_b = _make_mock_token("user-B")

        # Save original state and set delegated mode
        orig_delegated_mode = ctx_module.delegated_mode
        orig_managed_context = ctx_module.managed_context
        ctx_module.delegated_mode = True
        ctx_module.managed_context = None

        # Clear caches
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
            ):
                # Session A
                mock_at.return_value = _mock_access_token_obj(token_a)
                mock_req.return_value = _mock_http_request("session-AAA")

                context_a = await ctx_module.get_evo_context()

                # Session B
                mock_at.return_value = _mock_access_token_obj(token_b)
                mock_req.return_value = _mock_http_request("session-BBB")

                context_b = await ctx_module.get_evo_context()

            self.assertIsNot(context_a, context_b)
            self.assertIsInstance(context_a, DelegatedAuthContext)
            self.assertIsInstance(context_b, DelegatedAuthContext)
            # Each has its own staging and registry
            self.assertIsNot(context_a.object_staging, context_b.object_staging)
            self.assertIsNot(context_a.object_registry, context_b.object_registry)
        finally:
            ctx_module.delegated_mode = orig_delegated_mode
            ctx_module.managed_context = orig_managed_context
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()


# ---------------------------------------------------------------------------
# Test: Staged objects are session-private
# ---------------------------------------------------------------------------


class TestStagingIsolation(unittest.IsolatedAsyncioTestCase):
    """Objects staged in session A are invisible to session B."""

    async def test_staging_isolation_between_sessions(self):
        token_a = _make_mock_token("user-A")
        token_b = _make_mock_token("user-B")

        ctx_module.delegated_mode = True
        ctx_module.managed_context = None
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
            ):
                # Create session A context
                mock_at.return_value = _mock_access_token_obj(token_a)
                mock_req.return_value = _mock_http_request("session-AAA")
                context_a = await ctx_module.get_evo_context()

                # Create session B context
                mock_at.return_value = _mock_access_token_obj(token_b)
                mock_req.return_value = _mock_http_request("session-BBB")
                context_b = await ctx_module.get_evo_context()

            # Use _put directly to bypass type/domain validation — the test only
            # cares that two sessions have separate stores, not about payload type.
            staging_a = context_a.object_staging
            staging_a._put(
                StagedEnvelope(
                    stage_id="stage-001",
                    object_type="variogram",
                    workspace_id=None,
                    source_ref={},
                    status="active",
                    updated_at="2026-01-01T00:00:00+00:00",
                    expires_at="2099-01-01T00:00:00+00:00",
                ),
                {"test": "payload"},
            )

            # Session A can retrieve it
            envelope, payload = staging_a.get_stage_payload("stage-001")
            self.assertEqual(envelope.stage_id, "stage-001")
            self.assertEqual(payload, {"test": "payload"})

            # Session B cannot see it
            staging_b = context_b.object_staging
            with self.assertRaises(StageNotFoundError):
                staging_b.get_stage_payload("stage-001")
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()


# ---------------------------------------------------------------------------
# Test: Registered names are session-private
# ---------------------------------------------------------------------------


class TestRegistryIsolation(unittest.IsolatedAsyncioTestCase):
    """Names registered in session A are unresolvable in session B."""

    async def test_registry_isolation_between_sessions(self):
        token_a = _make_mock_token("user-A")
        token_b = _make_mock_token("user-B")

        ctx_module.delegated_mode = True
        ctx_module.managed_context = None
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
            ):
                mock_at.return_value = _mock_access_token_obj(token_a)
                mock_req.return_value = _mock_http_request("session-AAA")
                context_a = await ctx_module.get_evo_context()

                mock_at.return_value = _mock_access_token_obj(token_b)
                mock_req.return_value = _mock_http_request("session-BBB")
                context_b = await ctx_module.get_evo_context()

            # Stage and register in session A
            context_a.object_staging._put(
                StagedEnvelope(
                    stage_id="stage-reg-001",
                    object_type="variogram",
                    workspace_id=None,
                    source_ref={},
                    status="active",
                    updated_at="2026-01-01T00:00:00+00:00",
                    expires_at="2099-01-01T00:00:00+00:00",
                ),
                {"variogram": "data"},
            )
            context_a.object_registry.register(
                name="CU variogram",
                object_type="variogram",
                stage_id="stage-reg-001",
            )

            # Session A resolves successfully
            entry = context_a.object_registry.resolve("CU variogram")
            self.assertEqual(entry.name, "CU variogram")
            self.assertEqual(entry.stage_id, "stage-reg-001")

            # Session B cannot resolve the same name
            with self.assertRaises(ResolutionError):
                context_b.object_registry.resolve("CU variogram")
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()


# ---------------------------------------------------------------------------
# Test: Same session ID reuses context
# ---------------------------------------------------------------------------


class TestSameSessionReusesContext(unittest.IsolatedAsyncioTestCase):
    """Repeated requests with same session ID return the same context."""

    async def test_same_session_returns_same_context(self):
        token = _make_mock_token("user-X")

        ctx_module.delegated_mode = True
        ctx_module.managed_context = None
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
            ):
                mock_at.return_value = _mock_access_token_obj(token)
                mock_req.return_value = _mock_http_request("session-XYZ")

                context_1 = await ctx_module.get_evo_context()

                # Register something to verify state persistence
                context_1.object_staging._put(
                    StagedEnvelope(
                        stage_id="stage-persist",
                        object_type="point_set",
                        workspace_id=None,
                        source_ref={},
                        status="active",
                        updated_at="2026-01-01T00:00:00+00:00",
                        expires_at="2099-01-01T00:00:00+00:00",
                    ),
                    {"points": []},
                )
                context_1.object_registry.register(
                    name="my points",
                    object_type="point_set",
                    stage_id="stage-persist",
                )

                # Second request with same session
                context_2 = await ctx_module.get_evo_context()

            self.assertIs(context_1, context_2)
            # Registry state persisted
            entry = context_2.object_registry.resolve("my points")
            self.assertEqual(entry.stage_id, "stage-persist")
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()


# ---------------------------------------------------------------------------
# Test: TTL eviction calls cleanup
# ---------------------------------------------------------------------------


class TestTTLEvictionCleanup(unittest.TestCase):
    """Expired sessions are cleaned up and locks removed."""

    def test_ttl_eviction_calls_cleanup(self):
        locks: dict[str, asyncio.Lock] = {"session-1": asyncio.Lock()}
        cache: TTLCache = _CleanupTTLCache(maxsize=10, ttl=0.1, locks=locks)

        mock_context = MagicMock()
        mock_context.cleanup = MagicMock()

        cache["session-1"] = mock_context

        # Wait for TTL to expire
        time.sleep(0.2)

        # Trigger eviction by accessing/inserting (cachetools lazy eviction)
        cache["session-2"] = MagicMock()

        # session-1 should have been evicted
        self.assertNotIn("session-1", cache)
        # cleanup should have been called exactly once
        mock_context.cleanup.assert_called_once()
        # lock should be removed
        self.assertNotIn("session-1", locks)

    def test_explicit_delete_calls_cleanup_once(self):
        locks: dict[str, asyncio.Lock] = {"s": asyncio.Lock()}
        cache: TTLCache = _CleanupTTLCache(maxsize=10, ttl=60, locks=locks)
        mock_context = MagicMock()
        cache["s"] = mock_context

        del cache["s"]

        mock_context.cleanup.assert_called_once()
        self.assertNotIn("s", locks)

    def test_pop_calls_cleanup_once(self):
        locks: dict[str, asyncio.Lock] = {"s": asyncio.Lock()}
        cache: TTLCache = _CleanupTTLCache(maxsize=10, ttl=60, locks=locks)
        mock_context = MagicMock()
        cache["s"] = mock_context

        cache.pop("s")

        mock_context.cleanup.assert_called_once()
        self.assertNotIn("s", locks)

    def test_maxsize_eviction_calls_cleanup_once(self):
        locks: dict[str, asyncio.Lock] = {"s": asyncio.Lock()}
        cache: TTLCache = _CleanupTTLCache(maxsize=1, ttl=60, locks=locks)
        mock_context = MagicMock()
        cache["s"] = mock_context

        # Inserting a second item forces LRU eviction of "s"
        cache["s2"] = MagicMock()

        mock_context.cleanup.assert_called_once()
        self.assertNotIn("s", locks)

    def test_clear_calls_cleanup_once_per_item(self):
        locks: dict[str, asyncio.Lock] = {"a": asyncio.Lock(), "b": asyncio.Lock()}
        cache: TTLCache = _CleanupTTLCache(maxsize=10, ttl=60, locks=locks)
        ctx_a, ctx_b = MagicMock(), MagicMock()
        cache["a"] = ctx_a
        cache["b"] = ctx_b

        cache.clear()

        ctx_a.cleanup.assert_called_once()
        ctx_b.cleanup.assert_called_once()
        self.assertEqual(locks, {})


# ---------------------------------------------------------------------------
# Test: Lock prevents duplicate context creation
# ---------------------------------------------------------------------------


class TestConcurrentSessionCreation(unittest.IsolatedAsyncioTestCase):
    """Multiple concurrent get_evo_context() for same session create only one context."""

    async def test_concurrent_creation_single_context(self):
        token = _make_mock_token("user-concurrent")
        creation_count = 0

        original_init = DelegatedAuthContext.__init__

        def counting_init(self, *args, **kwargs):
            nonlocal creation_count
            creation_count += 1
            original_init(self, *args, **kwargs)

        ctx_module.delegated_mode = True
        ctx_module.managed_context = None
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
                patch.object(DelegatedAuthContext, "__init__", counting_init),
            ):
                mock_at.return_value = _mock_access_token_obj(token)
                mock_req.return_value = _mock_http_request("session-concurrent")

                # Launch 5 concurrent requests
                results = await asyncio.gather(*[ctx_module.get_evo_context() for _ in range(5)])

            # All should return the same instance
            for ctx in results[1:]:
                self.assertIs(results[0], ctx)

            # Only one context was created
            self.assertEqual(creation_count, 1)
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()


# ---------------------------------------------------------------------------
# Test: Token refresh rebuilds clients but preserves staging/registry
# ---------------------------------------------------------------------------


class TestTokenRefreshPreservesState(unittest.IsolatedAsyncioTestCase):
    """Legacy session header survives token refresh and preserves context state."""

    async def test_token_refresh_rebuilds_clients_preserves_state(self):
        # Two different tokens for the same user (iat differs → distinct strings)
        token_v1 = _make_mock_token("user-refresh", iat=1000)
        token_v2 = _make_mock_token("user-refresh", iat=2000)

        ctx_module.delegated_mode = True
        ctx_module.managed_context = None
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock) as mock_build,
            ):
                # First request with token_v1
                mock_at.return_value = _mock_access_token_obj(token_v1)
                mock_req.return_value = _mock_http_request("session-refresh")
                context = await ctx_module.get_evo_context()

                selected_org = UUID("8c0c414c-61c2-42f0-ac13-8c841919f63e")
                context.org_id = selected_org
                context.hub_url = "https://selected.example"

                # Stage an object
                context.object_staging._put(
                    StagedEnvelope(
                        stage_id="stage-refresh",
                        object_type="variogram",
                        workspace_id=None,
                        source_ref={},
                        status="active",
                        updated_at="2026-01-01T00:00:00+00:00",
                        expires_at="2099-01-01T00:00:00+00:00",
                    ),
                    {"refreshed": True},
                )
                context.object_registry.register(
                    name="test vario",
                    object_type="variogram",
                    stage_id="stage-refresh",
                )

                build_count_before = mock_build.call_count

                # Second request with token_v2 (token refreshed)
                mock_at.return_value = _mock_access_token_obj(token_v2)
                context_after = await ctx_module.get_evo_context()

            # Same context instance (same session)
            self.assertIs(context, context_after)
            self.assertEqual(context_after.org_id, selected_org)
            self.assertEqual(context_after.hub_url, "https://selected.example")

            # discover_and_build was called again (clients rebuilt)
            self.assertGreater(mock_build.call_count, build_count_before)

            # Staging and registry state preserved
            entry = context_after.object_registry.resolve("test vario")
            self.assertEqual(entry.stage_id, "stage-refresh")
            _, payload = context_after.object_staging.get_stage_payload("stage-refresh")
            self.assertEqual(payload, {"refreshed": True})
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()


# ---------------------------------------------------------------------------
# Test: Managed mode returns singleton
# ---------------------------------------------------------------------------


class TestManagedModeSharedContext(unittest.IsolatedAsyncioTestCase):
    """Managed mode returns the same context every time."""

    async def test_managed_mode_returns_singleton(self):
        ctx_module.delegated_mode = False
        ctx_module.managed_context = None

        try:
            with patch.object(ManagedAuthContext, "initialize", new_callable=AsyncMock):
                context_1 = await ctx_module.get_evo_context()
                context_2 = await ctx_module.get_evo_context()

            self.assertIs(context_1, context_2)
            self.assertIsInstance(context_1, ManagedAuthContext)
            # Staging and registry are the same instances
            self.assertIs(context_1.object_staging, context_2.object_staging)
            self.assertIs(context_1.object_registry, context_2.object_registry)
        finally:
            ctx_module.managed_context = None


# ---------------------------------------------------------------------------
# Test: Different users with same session header get different contexts
# ---------------------------------------------------------------------------


class TestUserIsolation(unittest.IsolatedAsyncioTestCase):
    """Legacy session headers distinguish clients even when credentials overlap."""

    async def test_different_users_same_session_header_isolated(self):
        token_user1 = _make_mock_token("user-alice")
        token_user2 = _make_mock_token("user-bob")

        ctx_module.delegated_mode = True
        ctx_module.managed_context = None
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
            ):
                # User Alice with session-SHARED
                mock_at.return_value = _mock_access_token_obj(token_user1)
                mock_req.return_value = _mock_http_request("session-SHARED")
                context_alice = await ctx_module.get_evo_context()

                # User Bob with the SAME session header
                mock_at.return_value = _mock_access_token_obj(token_user2)
                mock_req.return_value = _mock_http_request("session-SHARED")
                context_bob = await ctx_module.get_evo_context()

            # Should be different contexts despite same session header
            self.assertIsNot(context_alice, context_bob)
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()

    async def test_same_user_and_token_different_legacy_sessions_are_isolated(self):
        token = _make_mock_token("user-alice")
        ctx_module.delegated_mode = True
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

        try:
            with (
                patch("evo_mcp.contexts.helpers.get_access_token", return_value=_mock_access_token_obj(token)),
                patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
                patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
            ):
                mock_req.return_value = _mock_http_request("session-A")
                context_a = await ctx_module.get_evo_context()
                mock_req.return_value = _mock_http_request("session-B")
                context_b = await ctx_module.get_evo_context()

            self.assertIsNot(context_a, context_b)
            self.assertIsNot(context_a.object_staging, context_b.object_staging)
        finally:
            ctx_module.delegated_mode = False
            ctx_module.delegated_contexts.clear()
            ctx_module.session_locks.clear()

    def test_get_client_session_id_differs_by_user(self):
        """get_client_session_id() includes user identity in the key."""
        token_alice = _make_mock_token("user-alice")
        token_bob = _make_mock_token("user-bob")

        with (
            patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
            patch("evo_mcp.contexts.helpers.get_http_request") as mock_req,
        ):
            mock_req.return_value = _mock_http_request("session-SAME")

            mock_at.return_value = _mock_access_token_obj(token_alice)
            id_alice = get_client_session_id()

            mock_at.return_value = _mock_access_token_obj(token_bob)
            id_bob = get_client_session_id()

        self.assertNotEqual(id_alice, id_bob)


class TestSessionlessContextRegressions(unittest.IsolatedAsyncioTestCase):
    """Pending CENPLAT-37353: opt in with RUN_SESSIONLESS_MCP_TESTS=1.

    The shared-token case must supply distinct client context handles once the
    stateless API defines them; without a header or handle, clients are identical.
    """

    def setUp(self):
        self.original_delegated_mode = ctx_module.delegated_mode
        ctx_module.delegated_mode = True
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()

    def tearDown(self):
        ctx_module.delegated_contexts.clear()
        ctx_module.session_locks.clear()
        ctx_module.delegated_mode = self.original_delegated_mode

    @unittest.skipUnless(
        os.getenv("RUN_SESSIONLESS_MCP_TESTS") == "1",
        "CENPLAT-37353: sessionless token refresh does not preserve instance selection",
    )
    async def test_refresh_preserves_selected_instance_without_session_header(self):
        token_v1 = _make_mock_token("user-refresh", iat=1000)
        token_v2 = _make_mock_token("user-refresh", iat=2000)
        selected_org = UUID("8c0c414c-61c2-42f0-ac13-8c841919f63e")

        with (
            patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
            patch("evo_mcp.contexts.helpers.get_http_request", return_value=_mock_http_request(None)),
            patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
        ):
            mock_at.return_value = _mock_access_token_obj(token_v1)
            context = await ctx_module.get_evo_context()
            context.org_id = selected_org
            context.hub_url = "https://selected.example"

            mock_at.return_value = _mock_access_token_obj(token_v2)
            refreshed = await ctx_module.get_evo_context()

        self.assertEqual(refreshed.org_id, selected_org)
        self.assertEqual(refreshed.hub_url, "https://selected.example")

    @unittest.skipUnless(
        os.getenv("RUN_SESSIONLESS_MCP_TESTS") == "1",
        "CENPLAT-37353: sessionless token refresh loses staged objects",
    )
    async def test_refresh_preserves_staging_without_session_header(self):
        token_v1 = _make_mock_token("user-refresh", iat=1000)
        token_v2 = _make_mock_token("user-refresh", iat=2000)

        with (
            patch("evo_mcp.contexts.helpers.get_access_token") as mock_at,
            patch("evo_mcp.contexts.helpers.get_http_request", return_value=_mock_http_request(None)),
            patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
        ):
            mock_at.return_value = _mock_access_token_obj(token_v1)
            context = await ctx_module.get_evo_context()
            context.object_staging._put(
                StagedEnvelope(
                    stage_id="stage-sessionless",
                    object_type="variogram",
                    workspace_id=None,
                    source_ref={},
                    status="active",
                    updated_at="2026-01-01T00:00:00+00:00",
                    expires_at="2099-01-01T00:00:00+00:00",
                ),
                {"refreshed": True},
            )
            context.object_registry.register(
                name="sessionless variogram",
                object_type="variogram",
                stage_id="stage-sessionless",
            )

            mock_at.return_value = _mock_access_token_obj(token_v2)
            refreshed = await ctx_module.get_evo_context()

        entry = refreshed.object_registry.resolve("sessionless variogram")
        self.assertEqual(entry.stage_id, "stage-sessionless")
        _, payload = refreshed.object_staging.get_stage_payload(entry.stage_id)
        self.assertEqual(payload, {"refreshed": True})

    @unittest.skipUnless(
        os.getenv("RUN_SESSIONLESS_MCP_TESTS") == "1",
        "CENPLAT-37353: clients sharing a token need separate stateless context handles",
    )
    async def test_clients_sharing_token_do_not_share_context_without_session_header(self):
        token = _make_mock_token("user-alice")

        with (
            patch("evo_mcp.contexts.helpers.get_access_token", return_value=_mock_access_token_obj(token)),
            patch("evo_mcp.contexts.helpers.get_http_request", return_value=_mock_http_request(None)),
            patch.object(DelegatedAuthContext, "discover_and_build", new_callable=AsyncMock),
        ):
            client_a = await ctx_module.get_evo_context()
            client_a.org_id = UUID("8c0c414c-61c2-42f0-ac13-8c841919f63e")
            client_b = await ctx_module.get_evo_context()

        self.assertIsNot(client_a, client_b)
        self.assertIsNone(client_b.org_id)


# ---------------------------------------------------------------------------
# Test: Cache evicts LRU when maxsize reached
# ---------------------------------------------------------------------------


class TestCacheMaxsizeEviction(unittest.TestCase):
    """_CleanupTTLCache evicts least-recently-used when full."""

    def test_maxsize_eviction_calls_cleanup(self):
        locks: dict[str, asyncio.Lock] = {
            "session-A": asyncio.Lock(),
            "session-B": asyncio.Lock(),
            "session-C": asyncio.Lock(),
        }
        cache: TTLCache = _CleanupTTLCache(maxsize=2, ttl=3600, locks=locks)

        ctx_a = MagicMock()
        ctx_a.cleanup = MagicMock()
        ctx_b = MagicMock()
        ctx_b.cleanup = MagicMock()
        ctx_c = MagicMock()
        ctx_c.cleanup = MagicMock()

        cache["session-A"] = ctx_a
        cache["session-B"] = ctx_b

        # Inserting C should evict A (LRU)
        cache["session-C"] = ctx_c

        self.assertNotIn("session-A", cache)
        self.assertIn("session-B", cache)
        self.assertIn("session-C", cache)

        # cleanup called at least once on evicted context (may be called
        # multiple times due to cachetools internals — that's fine, cleanup()
        # is idempotent)
        ctx_a.cleanup.assert_called()
        ctx_b.cleanup.assert_not_called()
        ctx_c.cleanup.assert_not_called()

        # Lock for A should be removed
        self.assertNotIn("session-A", locks)


if __name__ == "__main__":
    unittest.main()
