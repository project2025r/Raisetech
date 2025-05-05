# Deployment Guide for Docker Setup

This guide provides instructions for deploying the Road AI Safety Enhancement application using Docker on an EC2 Ubuntu instance.

## Prerequisites

- Ubuntu EC2 instance with sufficient resources (recommended: 4GB RAM, 2 vCPUs)
- Docker and Docker Compose installed on the instance

## Docker Installation on Ubuntu

```bash
# Update package list
sudo apt-get update

# Install required packages
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update package list again
sudo apt-get update

# Install Docker CE
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to the docker group to run Docker without sudo
sudo usermod -aG docker $USER

# Apply the group changes (or log out and back in)
newgrp docker
```

## Environment Variables Setup

Before deploying, create environment files for both the backend and frontend:

### Backend Environment (.env file in backend directory)

```
# MongoDB Configuration
MONGO_URI=mongodb://mongo:27017
DB_NAME=LTA

# Backend Configuration
FLASK_APP=app.py
FLASK_ENV=production
PYTHONUNBUFFERED=1
```

### Frontend Environment (if needed)

You can pass environment variables to the frontend build process by adding them to the docker-compose.yml under the frontend service:

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      - REACT_APP_API_URL=http://your-ec2-public-ip
      - REACT_APP_BACKEND_PORT=5000
```

## Deployment Steps

1. Clone the repository to your EC2 instance:
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. Create the environment files as described above.

3. Start the application using Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. Monitor the logs:
   ```bash
   docker-compose logs -f
   ```

5. Access the application:
   - Frontend: `http://your-ec2-public-ip`
   - Backend API: `http://your-ec2-public-ip/api`

## Security Considerations

- Configure your EC2 security group to only allow traffic on the required ports (80 for the frontend, 5000 for the backend if direct access is needed)
- Set up SSL/TLS using a reverse proxy like Nginx for production deployments
- Use Docker secrets or a secure environment management system instead of .env files for sensitive information in production

## Troubleshooting

If you encounter any issues:

1. Check container status:
   ```bash
   docker-compose ps
   ```

2. View container logs:
   ```bash
   docker-compose logs backend
   docker-compose logs frontend
   docker-compose logs mongo
   ```

3. Access a container's shell:
   ```bash
   docker-compose exec backend /bin/bash
   docker-compose exec mongo mongosh
   ```

## Data Persistence

MongoDB data is persisted through a Docker volume named `mongo-data`. To back up this data:

```bash
docker run --rm -v mongo-data:/data -v $(pwd)/backup:/backup ubuntu tar -czvf /backup/mongo-data.tar.gz /data
```

To restore from a backup:

```bash
docker run --rm -v mongo-data:/data -v $(pwd)/backup:/backup ubuntu tar -xzvf /backup/mongo-data.tar.gz -C /
```

## Production Considerations

1. **Resource Monitoring**: Set up monitoring for your EC2 instance using CloudWatch.

2. **Scaling**: For increased traffic, consider scaling horizontally by deploying multiple instances behind a load balancer.

3. **Backups**: Implement regular automated backups for the MongoDB data.

4. **HTTPS**: Configure SSL/TLS for secure connections using a certificate from a trusted authority.

5. **Release Management**: Implement a CI/CD pipeline for automated deployments. 