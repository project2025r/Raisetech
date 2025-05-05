# Docker Setup for Road AI Safety Enhancement

This repository contains Docker configurations for deploying the Road AI Safety Enhancement application.

## Project Structure

```
.
├── backend/                 # Flask backend
│   ├── Dockerfile           # Backend container configuration
│   ├── .dockerignore        # Files to ignore in Docker build
│   └── ...
├── frontend/                # React.js frontend
│   ├── Dockerfile           # Frontend container configuration
│   ├── nginx.conf           # Nginx configuration for frontend
│   ├── .dockerignore        # Files to ignore in Docker build
│   └── ...
├── docker-compose.yml       # Multi-container Docker configuration
└── DEPLOYMENT.md            # Detailed deployment instructions
```

## Quick Start

1. Make sure Docker and Docker Compose are installed on your system.

2. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. Create necessary environment files as described in DEPLOYMENT.md.

4. Start the application:
   ```bash
   docker-compose up -d
   ```

5. Access the application:
   - Frontend: http://localhost
   - Backend API: http://localhost/api

## Container Architecture

The application uses three main containers:

1. **MongoDB (mongo)** - Database container running MongoDB 6.
   - Port: 27017
   - Persistent storage through Docker volume

2. **Backend (backend)** - Flask application container.
   - Port: 5000
   - Connects to MongoDB for data storage
   - Built using Python 3.11

3. **Frontend (frontend)** - React.js application with Nginx.
   - Port: 80
   - Communicates with backend through Nginx reverse proxy
   - Built using Node.js 18 and served with Nginx

All containers are connected through a Docker network named 'lta-network'.

## Environment Variables

The application uses environment variables for configuration. See DEPLOYMENT.md for details on required variables and how to set them up.

## For Detailed Deployment Instructions

See [DEPLOYMENT.md](./DEPLOYMENT.md) for complete deployment instructions, including:

- EC2 Ubuntu setup
- Environment configuration
- Security considerations
- Troubleshooting tips
- Production deployment best practices 