#!/usr/bin/env python3
# Copyright (c) 2025 Pranav Jadhav. All rights reserved.
# AI Agent Orchestration Platform - Deployment Script

import os
import sys
import asyncio
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def run_command(command: str, check: bool = True) -> Tuple[int, str]:
    """Run a shell command and return exit code and output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout + e.stderr
    except Exception as e:
        return 1, str(e)

class PlatformDeployer:
    """Complete platform deployment and management"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.services = [
            "api-gateway",
            "agent-registry", 
            "orchestration-engine",
            "memory-management",
            "hitl-service"
        ]
        self.health_urls = {
            "api-gateway": "http://localhost:8000/health",
            "agent-registry": "http://localhost:8001/health",
            "orchestration-engine": "http://localhost:8002/health",
            "memory-management": "http://localhost:8003/health",
            "hitl-service": "http://localhost:8004/health",
        }
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        print_header("🔍 Checking Prerequisites")
        
        prerequisites = {
            "python": "python --version",
            "node": "node --version",
            "npm": "npm --version",
            "docker": "docker --version",
            "docker-compose": "docker-compose --version",
            "git": "git --version"
        }
        
        all_good = True
        for name, command in prerequisites.items():
            exit_code, output = run_command(command, check=False)
            if exit_code == 0:
                version = output.strip().split('\n')[0]
                print_success(f"{name}: {version}")
            else:
                print_error(f"{name} is not installed or not in PATH")
                all_good = False
        
        return all_good
    
    def setup_environment(self) -> bool:
        """Set up the development environment"""
        print_header("🌍 Setting up Environment")
        
        # Check if .env exists
        env_file = self.project_root / ".env"
        env_template = self.project_root / ".env.template"
        
        if not env_file.exists() and env_template.exists():
            print_info("Creating .env file from template...")
            env_file.write_text(env_template.read_text())
            print_success("Created .env file")
            print_warning("Please edit .env file with your configuration before proceeding")
            return False
        elif env_file.exists():
            print_success(".env file already exists")
        else:
            print_error("No .env.template found")
            return False
        
        # Create Python virtual environment if it doesn't exist
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            print_info("Creating Python virtual environment...")
            exit_code, output = run_command("python -m venv venv")
            if exit_code == 0:
                print_success("Virtual environment created")
            else:
                print_error(f"Failed to create virtual environment: {output}")
                return False
        
        # Install Python dependencies
        if sys.platform.startswith('win'):
            pip_command = "venv\\Scripts\\pip"
        else:
            pip_command = "venv/bin/pip"
        
        print_info("Installing Python dependencies...")
        exit_code, output = run_command(f"{pip_command} install -r requirements.txt")
        if exit_code == 0:
            print_success("Python dependencies installed")
        else:
            print_error(f"Failed to install Python dependencies: {output}")
            return False
        
        # Install frontend dependencies
        web_ui_path = self.project_root / "web-ui"
        if web_ui_path.exists():
            print_info("Installing frontend dependencies...")
            exit_code, output = run_command("npm install", cwd=web_ui_path)
            if exit_code == 0:
                print_success("Frontend dependencies installed")
            else:
                print_error(f"Failed to install frontend dependencies: {output}")
                return False
        
        return True
    
    def start_infrastructure(self) -> bool:
        """Start infrastructure services with Docker Compose"""
        print_header("🚀 Starting Infrastructure Services")
        
        # Start PostgreSQL, Redis, and ChromaDB
        services = ["postgres", "redis", "chroma"]
        
        for service in services:
            print_info(f"Starting {service}...")
            exit_code, output = run_command(f"docker-compose up -d {service}")
            if exit_code == 0:
                print_success(f"{service} started")
            else:
                print_error(f"Failed to start {service}: {output}")
                return False
        
        # Wait for services to be ready
        print_info("Waiting for services to be ready...")
        time.sleep(15)
        
        return True
    
    def initialize_database(self) -> bool:
        """Initialize the database"""
        print_header("🗄️ Initializing Database")
        
        if sys.platform.startswith('win'):
            python_command = "venv\\Scripts\\python"
        else:
            python_command = "venv/bin/python"
        
        print_info("Running database initialization...")
        exit_code, output = run_command(f"{python_command} scripts/migration/init_db.py")
        if exit_code == 0:
            print_success("Database initialized successfully")
            return True
        else:
            print_error(f"Failed to initialize database: {output}")
            return False
    
    def build_frontend(self) -> bool:
        """Build the frontend application"""
        print_header("🏗️ Building Frontend")
        
        web_ui_path = self.project_root / "web-ui"
        if not web_ui_path.exists():
            print_warning("Frontend directory not found, skipping...")
            return True
        
        print_info("Building React frontend...")
        exit_code, output = run_command("npm run build", cwd=web_ui_path)
        if exit_code == 0:
            print_success("Frontend built successfully")
            return True
        else:
            print_error(f"Failed to build frontend: {output}")
            return False
    
    def start_backend_services(self) -> bool:
        """Start all backend services"""
        print_header("⚙️ Starting Backend Services")
        
        if sys.platform.startswith('win'):
            python_command = "venv\\Scripts\\python"
        else:
            python_command = "venv/bin/python"
        
        # Start services in background
        processes = []
        for service in self.services:
            service_path = self.project_root / "services" / service / "app"
            if service_path.exists():
                print_info(f"Starting {service}...")
                
                # Determine port
                ports = {
                    "api-gateway": 8000,
                    "agent-registry": 8001,
                    "orchestration-engine": 8002,
                    "memory-management": 8003,
                    "hitl-service": 8004
                }
                port = ports.get(service, 8000)
                
                # Start service
                cmd = f"{python_command} -m uvicorn app.main:app --host 0.0.0.0 --port {port}"
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    cwd=service_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                processes.append((service, proc))
                
                print_success(f"{service} starting on port {port}")
        
        # Wait a moment for services to start
        time.sleep(10)
        
        return True
    
    def run_health_checks(self) -> bool:
        """Run health checks on all services"""
        print_header("🏥 Running Health Checks")
        
        all_healthy = True
        
        for service, url in self.health_urls.items():
            print_info(f"Checking {service}...")
            
            try:
                import requests
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print_success(f"{service} is healthy")
                else:
                    print_error(f"{service} returned status {response.status_code}")
                    all_healthy = False
            except ImportError:
                # Use curl if requests is not available
                exit_code, output = run_command(f"curl -f {url}", check=False)
                if exit_code == 0:
                    print_success(f"{service} is healthy")
                else:
                    print_error(f"{service} health check failed")
                    all_healthy = False
            except Exception as e:
                print_error(f"{service} health check failed: {e}")
                all_healthy = False
        
        return all_healthy
    
    def run_tests(self) -> bool:
        """Run the test suite"""
        print_header("🧪 Running Tests")
        
        if sys.platform.startswith('win'):
            python_command = "venv\\Scripts\\python"
        else:
            python_command = "venv/bin/python"
        
        # Run backend tests
        print_info("Running backend tests...")
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            exit_code, output = run_command(f"{python_command} -m pytest tests/ -v")
            if exit_code == 0:
                print_success("Backend tests passed")
            else:
                print_warning(f"Some backend tests failed: {output}")
        else:
            print_warning("No tests directory found, skipping tests")
        
        # Run frontend tests if available
        web_ui_path = self.project_root / "web-ui"
        if web_ui_path.exists():
            print_info("Running frontend tests...")
            exit_code, output = run_command("npm test -- --watchAll=false", cwd=web_ui_path)
            if exit_code == 0:
                print_success("Frontend tests passed")
            else:
                print_warning(f"Some frontend tests failed: {output}")
        
        return True
    
    def generate_deployment_report(self) -> None:
        """Generate deployment report"""
        print_header("📋 Deployment Report")
        
        report = {
            "deployment_time": datetime.now().isoformat(),
            "environment": self.environment,
            "platform": sys.platform,
            "services": {}
        }
        
        # Check service status
        for service, url in self.health_urls.items():
            try:
                import requests
                response = requests.get(url, timeout=5)
                status = "healthy" if response.status_code == 200 else f"unhealthy ({response.status_code})"
            except:
                exit_code, _ = run_command(f"curl -f {url}", check=False)
                status = "healthy" if exit_code == 0 else "unhealthy"
            
            report["services"][service] = {
                "status": status,
                "url": url
            }
        
        # Save report
        report_path = self.project_root / "deployment-report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print_success(f"Deployment report saved to: {report_path}")
        
        # Print summary
        print_info("Service Status Summary:")
        for service, info in report["services"].items():
            if info["status"] == "healthy":
                print_success(f"  {service}: {info['status']} ({info['url']})")
            else:
                print_error(f"  {service}: {info['status']} ({info['url']})")
    
    async def deploy(self) -> bool:
        """Run complete deployment"""
        print_header("🚀 AI Agent Orchestration Platform Deployment")
        print_info(f"Environment: {self.environment}")
        print_info(f"Platform: {sys.platform}")
        print_info(f"Python: {sys.version}")
        
        steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Environment Setup", self.setup_environment),
            ("Infrastructure Services", self.start_infrastructure),
            ("Database Initialization", self.initialize_database),
            ("Frontend Build", self.build_frontend),
            ("Backend Services", self.start_backend_services),
            ("Health Checks", self.run_health_checks),
            ("Test Suite", self.run_tests),
        ]
        
        for step_name, step_func in steps:
            print_info(f"Running: {step_name}")
            try:
                success = step_func()
                if success:
                    print_success(f"✅ {step_name} completed")
                else:
                    print_error(f"❌ {step_name} failed")
                    
                    # Continue with deployment even if some steps fail
                    user_input = input(f"{Colors.WARNING}Continue with deployment? (y/n): {Colors.ENDC}")
                    if user_input.lower() != 'y':
                        print_error("Deployment aborted by user")
                        return False
            except Exception as e:
                print_error(f"❌ {step_name} failed with error: {e}")
                return False
        
        # Generate final report
        self.generate_deployment_report()
        
        print_header("🎉 Deployment Complete!")
        print_success("AI Agent Orchestration Platform is now running!")
        print_info("Access points:")
        print_info("  • API Gateway: http://localhost:8000")
        print_info("  • Web Dashboard: http://localhost:3000")
        print_info("  • API Documentation: http://localhost:8000/docs")
        print_info("  • Grafana Monitoring: http://localhost:3001")
        print_info("  • Prometheus Metrics: http://localhost:9090")
        
        print_warning("Next Steps:")
        print_info("  1. Configure your .env file with production values")
        print_info("  2. Set up SSL certificates for production")
        print_info("  3. Configure monitoring and alerting")
        print_info("  4. Review security settings")
        print_info("  5. Set up backup procedures")
        
        return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy AI Agent Orchestration Platform")
    parser.add_argument("--env", default="development", choices=["development", "staging", "production"],
                       help="Deployment environment")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--docker-only", action="store_true", help="Use only Docker deployment")
    
    args = parser.parse_args()
    
    deployer = PlatformDeployer(environment=args.env)
    
    try:
        success = asyncio.run(deployer.deploy())
        if success:
            print_success("🎉 Deployment completed successfully!")
            sys.exit(0)
        else:
            print_error("❌ Deployment failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print_warning("\n⚠️ Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"❌ Deployment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()