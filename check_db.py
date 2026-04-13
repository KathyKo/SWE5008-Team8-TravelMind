"""Quick DB inspector — run: python check_db.py"""
from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://travelmind_user:TravelMind_DB_2026!@travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com:5432/travelmind",
    pool_pre_ping=True
)

QUERIES = {
    "All tables + columns": """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """,
    "plans (all)": """
        SELECT plan_id, origin, destination, dates, budget, via_debate, created_at
        FROM plans ORDER BY created_at DESC;
    """,
    "plan_itineraries (count per plan)": """
        SELECT plan_id, COUNT(*) as options FROM plan_itineraries GROUP BY plan_id;
    """,
    "plan_flights (count per plan)": """
        SELECT plan_id, direction, COUNT(*) as count
        FROM plan_flights GROUP BY plan_id, direction ORDER BY plan_id;
    """,
    "plan_hotels (count per plan)": """
        SELECT plan_id, COUNT(*) as hotels FROM plan_hotels GROUP BY plan_id;
    """,
    "plan_explains (all)": """
        SELECT plan_id, created_at,
               LEFT(chain_of_thought, 100) as cot_preview
        FROM plan_explains ORDER BY created_at DESC;
    """,
}

with engine.connect() as conn:
    for title, sql in QUERIES.items():
        print(f"\n{'='*55}")
        print(f"  {title}")
        print(f"{'='*55}")
        rows = conn.execute(text(sql)).fetchall()
        if not rows:
            print("  (empty)")
        for row in rows:
            print(" ", dict(row._mapping))
