import unittest

from app.db.session import SessionLocal
from app.repositories.users import UserRepository
from app.schemas.query import QueryRequest
from app.services.query_service import QueryService


class TestHourRangeQuery(unittest.TestCase):
    def test_query_keeps_hour_window_filter(self) -> None:
        db = SessionLocal()
        try:
            user = UserRepository(db).get_by_email("business@demo.local")
            self.assertIsNotNone(user)

            result = QueryService(db).run(
                QueryRequest(question="Выручка с 6 часов до 18 часов 15 марта по часам"),
                user,
            )

            self.assertEqual(result.status, "executed")
            filter_pairs = [(item.key, item.operator, item.value) for item in result.query_plan.filters]
            self.assertIn(("order_hour", "gte", 6), filter_pairs)
            self.assertIn(("order_hour", "lte", 18), filter_pairs)
            self.assertIn("EXTRACT(HOUR FROM ot.order_timestamp)", result.generated_sql)
            self.assertIn(">=", result.generated_sql)
            self.assertIn("<=", result.generated_sql)
            if result.rows:
                hours = [row["order_hour"] for row in result.rows if "order_hour" in row]
                self.assertTrue(all(6 <= hour <= 18 for hour in hours))
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
