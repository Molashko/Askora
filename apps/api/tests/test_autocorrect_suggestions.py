import unittest

from app.db.session import SessionLocal
from app.repositories.users import UserRepository
from app.schemas.query import QueryRequest
from app.services.query_service import QueryService


class TestAutocorrectSuggestions(unittest.TestCase):
    def test_typo_query_keeps_metric_and_suggests_closer_word(self) -> None:
        db = SessionLocal()
        try:
            user = UserRepository(db).get_by_email("business@demo.local")
            self.assertIsNotNone(user)

            result = QueryService(db).run(QueryRequest(question="пожалуйста сколько атменилось за вчера"), user)

            self.assertEqual(result.status, "executed")
            self.assertIn("cancelled_orders", [item.key for item in result.query_plan.metrics])
            self.assertTrue(
                any("имелось в виду" in item.lower() for item in result.query_plan.warnings),
                msg=result.query_plan.warnings,
            )
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
