import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import auth_service


class AuthServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database_path = auth_service.DATABASE_PATH
        auth_service.DATABASE_PATH = Path(self.temp_dir.name) / "auth.sqlite3"
        auth_service._initialize_database()

        self.app = FastAPI()
        self.app.include_router(auth_service.auth_router)
        self.client = TestClient(self.app)

    def tearDown(self):
        auth_service.DATABASE_PATH = self.original_database_path
        self.temp_dir.cleanup()

    def test_register_login_me_and_logout(self):
        response = self.client.post(
            "/auth/register",
            json={"email": "User@Example.com", "password": "correct horse battery"}
        )
        self.assertEqual(response.status_code, 201)
        registration = response.json()
        self.assertEqual(registration["user"]["email"], "user@example.com")

        headers = {"Authorization": f"Bearer {registration['access_token']}"}
        self.assertEqual(self.client.get("/auth/me", headers=headers).status_code, 200)
        self.assertEqual(self.client.post("/auth/logout", headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/auth/me", headers=headers).status_code, 401)

    def test_duplicate_registration_and_invalid_login_are_rejected(self):
        credentials = {"email": "user@example.com", "password": "correct horse battery"}
        self.assertEqual(self.client.post("/auth/register", json=credentials).status_code, 201)
        self.assertEqual(self.client.post("/auth/register", json=credentials).status_code, 409)
        self.assertEqual(
            self.client.post(
                "/auth/login",
                json={"email": credentials["email"], "password": "wrong password"}
            ).status_code,
            401
        )

    def test_document_access_is_scoped_to_owner(self):
        owner = auth_service.register_user("owner@example.com", "correct horse battery")
        other_user = auth_service.register_user("other@example.com", "another secure phrase")
        auth_service.record_document_owner("doc-1", owner["id"])

        self.assertTrue(auth_service.user_owns_document("doc-1", owner["id"]))
        self.assertFalse(auth_service.user_owns_document("doc-1", other_user["id"]))

    def test_access_tokens_expire(self):
        user = auth_service.register_user("owner@example.com", "correct horse battery")
        token = auth_service.issue_access_token(user["id"])
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        with auth_service._connection() as connection:
            connection.execute(
                "UPDATE access_tokens SET expires_at = 0 WHERE token_hash = ?",
                (token_hash,)
            )

        self.assertIsNone(auth_service.resolve_access_token(token))


if __name__ == "__main__":
    unittest.main()