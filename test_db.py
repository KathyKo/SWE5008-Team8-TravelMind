from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://travelmind_user:TravelMind_DB_2026!@travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com:5432/travelmind",
    pool_pre_ping=True
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print("Connected:", result.fetchone())
