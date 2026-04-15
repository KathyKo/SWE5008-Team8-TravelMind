# TravelMind Database Setup (AWS RDS PostgreSQL)

This guide explains how team members can connect to the **TravelMind AWS
RDS PostgreSQL database** using:

-   **pgAdmin** (GUI database tool)
-   **Python** (backend development)

------------------------------------------------------------------------

# 1. Database Information

Database type:

PostgreSQL (AWS RDS)

Region:

ap-southeast-1

Database name:

travelmind

Port:

5432

Host (RDS endpoint):

travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com

DBPassword: 	

postgres_admin: 7DuGcoOCIa7plAsR3QFs

travelmind_user: TravelMind_DB_2026!

------------------------------------------------------------------------

# 2. Connecting via pgAdmin (Recommended for DB inspection)

## Step 1 --- Install pgAdmin

Download:

https://www.pgadmin.org/download/

Install the version for your OS.

------------------------------------------------------------------------

## Step 2 --- Register a new server

Open **pgAdmin**.

Left panel:

Servers\
→ Right click\
→ Register\
→ Server

------------------------------------------------------------------------

## Step 3 --- Fill General tab

Name:

TravelMind AWS

------------------------------------------------------------------------

## Step 4 --- Fill Connection tab

Host name/address:

travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com

Port:

5432

Maintenance database:

travelmind

Username:

`<your username>`{=html}

Password:

`<your password>`{=html}

Enable:

Save password

Click:

Save

------------------------------------------------------------------------

## Step 5 --- Verify connection

If successful you will see:

Servers\
└ TravelMind AWS\
└ Databases\
└ travelmind

You can inspect tables here:

Schemas\
→ public\
→ Tables

------------------------------------------------------------------------

# 3. Connecting via Python

Python backend uses **SQLAlchemy** with **psycopg2**.

------------------------------------------------------------------------

# Step 1 --- Install required libraries

Inside the project virtual environment:

pip install sqlalchemy psycopg2-binary python-dotenv

Optional but recommended:

pip install alembic

Final dependency list:

sqlalchemy\
psycopg2-binary\
python-dotenv

------------------------------------------------------------------------

# Step 2 --- Configure DATABASE_URL

Create a `.env` file in project root.

Example:

DATABASE_URL=postgresql://USERNAME:PASSWORD@travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com:5432/travelmind

Real credentials:

DATABASE_URL=postgresql://travelmind_user:TravelMind_DB_2026!@travelmind-db.cnsc26mwucgt.ap-southeast-1.rds.amazonaws.com:5432/travelmind

------------------------------------------------------------------------

# Step 3 --- SQLAlchemy connection code

Example:

backend/db/database.py

``` python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
```

------------------------------------------------------------------------

# Step 4 --- Test database connection

Create a test file:

backend/db/test_rds_connection.py

``` python
from sqlalchemy import text
from database import engine

with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    for row in result:
        print(row)
```

Run:

python backend/db/test_rds_connection.py

Expected output example:

PostgreSQL 15.x on x86_64

------------------------------------------------------------------------

# 4. Creating Tables

Tables are created using SQLAlchemy models.

Example:

``` python
from database import engine
from models import Base

Base.metadata.create_all(bind=engine)
```

This will automatically create tables in the **RDS database**.

------------------------------------------------------------------------

# 5. Recommended Development Workflow

Typical workflow:

Python backend\
↓\
SQLAlchemy ORM\
↓\
AWS RDS PostgreSQL\
↓\
Inspect tables using pgAdmin

pgAdmin is useful for:

-   checking tables
-   debugging data
-   running SQL queries

------------------------------------------------------------------------

# 6. Common Issues

## psycopg2 not installed

Error:

ModuleNotFoundError: No module named psycopg2

Fix:

pip install psycopg2-binary

------------------------------------------------------------------------

## Cannot connect to RDS

Possible reasons:

1.  security group blocking your IP
2.  incorrect username/password
3.  wrong database name

Contact the team lead if connection fails.

------------------------------------------------------------------------

# 7. Security Notes

Never commit credentials to Git.

`.env` should be ignored by git:

.env

Add to `.gitignore`:

.env

------------------------------------------------------------------------

# 8. Useful Tools

  Tool         Purpose
  ------------ -----------------------
  pgAdmin      Database GUI
  SQLAlchemy   ORM
  psycopg2     PostgreSQL driver
  dotenv       environment variables

------------------------------------------------------------------------

# 9. Architecture (TravelMind backend)

FastAPI Backend\
│\
│ SQLAlchemy\
│\
AWS RDS PostgreSQL\
│\
│\
pgAdmin (inspection/debug)
