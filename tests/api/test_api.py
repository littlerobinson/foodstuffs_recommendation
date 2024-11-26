import unittest

from fastapi.testclient import TestClient

from api.main import app


class TestLoadConfig(unittest.TestCase):
    def test_call_root(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Foodstuff Recommendation API !"}


if __name__ == "__main__":
    unittest.main()
