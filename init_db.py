"""Run once to create all tables on RDS."""
from backend.db.database import engine
from backend.db.models import Base

Base.metadata.create_all(bind=engine)
print("Tables created successfully.")
