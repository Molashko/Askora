import unittest

from app.db.session import SessionLocal
from app.repositories.users import UserRepository
from app.schemas.query import QueryRequest
from app.services.query_service import QueryService


class TestRankingQueries(unittest.TestCase):
    def test_max_day_revenue_query_uses_sort_and_limit(self) -> None:
        db = SessionLocal()
        try:
            user = UserRepository(db).get_by_email("business@demo.local")
            self.assertIsNotNone(user)

            result = QueryService(db).run(QueryRequest(question="День с самой большой выручкой за март"), user)

            self.assertEqual(result.status, "executed")
            self.assertEqual(result.query_plan.sort, "total_revenue DESC")
            self.assertEqual(result.query_plan.limit, 1)
            self.assertEqual([item.key for item in result.query_plan.dimensions], ["order_date"])
            self.assertEqual(result.row_count, 1)
        finally:
            db.close()

    def test_top3_city_query_uses_sort_and_limit(self) -> None:
        db = SessionLocal()
        try:
            user = UserRepository(db).get_by_email("business@demo.local")
            self.assertIsNotNone(user)

            result = QueryService(db).run(QueryRequest(question="Топ-3 города по выручке за март"), user)

            self.assertEqual(result.status, "executed")
            self.assertEqual(result.query_plan.sort, "total_revenue DESC")
            self.assertEqual(result.query_plan.limit, 3)
            self.assertEqual([item.key for item in result.query_plan.dimensions], ["city_id"])
            self.assertLessEqual(result.row_count, 3)
        finally:
            db.close()

    def test_worst_day_query_uses_ascending_sort(self) -> None:
        db = SessionLocal()
        try:
            user = UserRepository(db).get_by_email("business@demo.local")
            self.assertIsNotNone(user)

            result = QueryService(db).run(QueryRequest(question="Худший день по выручке за март"), user)

            self.assertEqual(result.status, "executed")
            self.assertEqual(result.query_plan.sort, "total_revenue ASC")
            self.assertEqual(result.query_plan.limit, 1)
            self.assertEqual([item.key for item in result.query_plan.dimensions], ["order_date"])
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
