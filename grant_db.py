"""
Run once with postgres_admin to grant schema privileges to travelmind_user,
then create all tables.
"""
from sqlalchemy import create_engine, text
from backend.db.models import Base

ADMIN_URL = "postgresql://postgres_admin:7DuGcoOCIa7plAsR3QFs@travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com:5432/travelmind"

admin_engine = create_engine(ADMIN_URL, pool_pre_ping=True)

with admin_engine.connect() as conn:
    conn.execute(text("GRANT USAGE ON SCHEMA public TO travelmind_user;"))
    conn.execute(text("GRANT CREATE ON SCHEMA public TO travelmind_user;"))
    conn.execute(text("GRANT ALL ON ALL TABLES IN SCHEMA public TO travelmind_user;"))
    conn.execute(text("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO travelmind_user;"))
    conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO travelmind_user;"))
    conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO travelmind_user;"))
    conn.commit()
    print("Grants applied.")

# Now create tables as admin (safer)
Base.metadata.create_all(bind=admin_engine)
print("Tables created successfully.")
