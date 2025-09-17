#!/usr/bin/env python3
# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Database Initialization

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from alembic import command
from alembic.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.models.database import Base
from shared.config.database import DatabaseConfig

class DatabaseInitializer:
    """Initialize and manage database setup"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        
    async def create_database_if_not_exists(self):
        """Create the database if it doesn't exist"""
        # Parse database URL to get connection details
        db_url = self.config.DATABASE_URL
        
        if db_url.startswith('postgresql://'):
            # Extract connection details
            import urllib.parse as urlparse
            parsed = urlparse.urlparse(db_url)
            
            # Connect to postgres db to create our target database
            admin_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/postgres"
            
            try:
                conn = await asyncpg.connect(admin_url)
                
                # Check if database exists
                db_name = parsed.path[1:]  # Remove leading slash
                exists = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1", db_name
                )
                
                if not exists:
                    print(f"Creating database: {db_name}")
                    await conn.execute(f'CREATE DATABASE "{db_name}"')
                    print(f"Database {db_name} created successfully!")
                else:
                    print(f"Database {db_name} already exists")
                    
                await conn.close()
                
            except Exception as e:
                print(f"Error creating database: {e}")
                raise
    
    def create_tables(self):
        """Create all tables using SQLAlchemy"""
        try:
            engine = create_engine(
                self.config.DATABASE_URL,
                echo=self.config.DATABASE_ECHO
            )
            
            print("Creating database tables...")
            Base.metadata.create_all(bind=engine)
            print("Database tables created successfully!")
            
            engine.dispose()
            
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise
    
    def setup_alembic(self):
        """Setup Alembic for database migrations"""
        try:
            # Setup alembic configuration
            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))
            alembic_cfg.set_main_option("sqlalchemy.url", self.config.DATABASE_URL)
            
            # Create alembic version table
            print("Setting up Alembic...")
            command.stamp(alembic_cfg, "head")
            print("Alembic setup complete!")
            
        except Exception as e:
            print(f"Error setting up Alembic: {e}")
            # Continue without Alembic for now
            pass
    
    def create_initial_data(self):
        """Create initial data for the platform"""
        try:
            engine = create_engine(
                self.config.DATABASE_URL,
                echo=self.config.DATABASE_ECHO
            )
            
            with engine.connect() as conn:
                # Create admin user if not exists
                admin_exists = conn.execute(text(
                    "SELECT 1 FROM users WHERE email = 'admin@ai-orchestration.local'"
                )).fetchone()
                
                if not admin_exists:
                    print("Creating admin user...")
                    from passlib.context import CryptContext
                    
                    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                    hashed_password = pwd_context.hash("admin123")  # Change in production!
                    
                    conn.execute(text("""
                        INSERT INTO users (id, email, username, full_name, hashed_password, role, is_active, is_verified)
                        VALUES (
                            gen_random_uuid(),
                            'admin@ai-orchestration.local',
                            'admin',
                            'System Administrator',
                            :password,
                            'admin',
                            true,
                            true
                        )
                    """), {"password": hashed_password})
                    
                    conn.commit()
                    print("Admin user created! Email: admin@ai-orchestration.local, Password: admin123")
                else:
                    print("Admin user already exists")
            
            engine.dispose()
            
        except Exception as e:
            print(f"Error creating initial data: {e}")
            pass  # Continue without initial data for now
    
    async def initialize(self):
        """Run complete database initialization"""
        try:
            print("=== AI Agent Orchestration Platform - Database Setup ===\n")
            
            # Step 1: Create database
            await self.create_database_if_not_exists()
            
            # Step 2: Create tables
            self.create_tables()
            
            # Step 3: Setup Alembic (optional)
            self.setup_alembic()
            
            # Step 4: Create initial data
            self.create_initial_data()
            
            print("\n=== Database initialization completed successfully! ===")
            print("\nNext steps:")
            print("1. Start the API Gateway: cd services/api-gateway && uvicorn app.main:app --reload")
            print("2. Start other services as needed")
            print("3. Access the API docs at: http://localhost:8000/docs")
            
        except Exception as e:
            print(f"Database initialization failed: {e}")
            sys.exit(1)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Initialize AI Agent Orchestration Platform Database')
    parser.add_argument('--force', action='store_true', help='Force recreate tables')
    parser.add_argument('--data-only', action='store_true', help='Only create initial data')
    
    args = parser.parse_args()
    
    if args.data_only:
        initializer = DatabaseInitializer()
        initializer.create_initial_data()
        return
    
    if args.force:
        print("WARNING: Force mode will drop and recreate all tables!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted")
            return
        
        # Drop all tables
        from sqlalchemy import create_engine
        config = DatabaseConfig()
        engine = create_engine(config.DATABASE_URL)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
    
    # Run initialization
    asyncio.run(DatabaseInitializer().initialize())

if __name__ == "__main__":
    main()