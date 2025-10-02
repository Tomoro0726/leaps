#!/bin/bash
# Deployment script for Sakura Internet High-Power Dok API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LEAPS - Sakura Dok Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if required environment variables are set
if [ -z "$SAKURA_API_TOKEN" ]; then
    echo -e "${RED}Error: SAKURA_API_TOKEN environment variable is not set${NC}"
    echo "Please set it with: export SAKURA_API_TOKEN='your-api-token'"
    exit 1
fi

if [ -z "$DATABASE_URL" ]; then
    echo -e "${RED}Error: DATABASE_URL environment variable is not set${NC}"
    exit 1
fi

if [ -z "$STORAGE_URL" ]; then
    echo -e "${RED}Error: STORAGE_URL environment variable is not set${NC}"
    exit 1
fi

if [ -z "$BLOB_READ_WRITE_TOKEN" ]; then
    echo -e "${RED}Error: BLOB_READ_WRITE_TOKEN environment variable is not set${NC}"
    exit 1
fi

# Configuration
CONTAINER_NAME="leaps-worker-$(date +%s)"
IMAGE=""
GPU_TYPE="A100"
GPU_COUNT=1
CPU="8"
MEMORY="32Gi"
API_ENDPOINT="https://api.koukaryoku-dok.sakura.ad.jp/v1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        -c|--cpu)
            CPU="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required environment variables:"
            echo "  SAKURA_API_TOKEN        Sakura Dok API token"
            echo "  DATABASE_URL            PostgreSQL connection string"
            echo "  STORAGE_URL             Vercel Blob Storage URL"
            echo "  BLOB_READ_WRITE_TOKEN   Vercel Blob access token"
            echo ""
            echo "Options:"
            echo "  -i, --image IMAGE       Docker image to deploy (required)"
            echo "  -n, --name NAME         Container name (default: leaps-worker-TIMESTAMP)"
            echo "  -g, --gpu TYPE          GPU type: A100, V100, T4 (default: A100)"
            echo "  --gpu-count COUNT       Number of GPUs (default: 1)"
            echo "  -c, --cpu CORES         CPU cores (default: 8)"
            echo "  -m, --memory SIZE       Memory size (default: 32Gi)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Example:"
            echo "  export SAKURA_API_TOKEN='your-token'"
            echo "  export DATABASE_URL='postgresql://...'"
            echo "  export STORAGE_URL='https://...'"
            echo "  export BLOB_READ_WRITE_TOKEN='vercel_blob_...'"
            echo "  $0 -i myregistry.example.com/leaps-worker:latest -g A100"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if image is specified
if [ -z "$IMAGE" ]; then
    echo -e "${RED}Error: Docker image not specified. Use -i option.${NC}"
    exit 1
fi

echo -e "${YELLOW}Deployment Configuration:${NC}"
echo -e "  Container name: ${BLUE}$CONTAINER_NAME${NC}"
echo -e "  Image: ${BLUE}$IMAGE${NC}"
echo -e "  GPU: ${BLUE}$GPU_TYPE x $GPU_COUNT${NC}"
echo -e "  CPU: ${BLUE}$CPU cores${NC}"
echo -e "  Memory: ${BLUE}$MEMORY${NC}"
echo ""

# Create JSON payload
PAYLOAD=$(cat <<EOF
{
  "name": "$CONTAINER_NAME",
  "image": "$IMAGE",
  "gpu": {
    "type": "$GPU_TYPE",
    "count": $GPU_COUNT
  },
  "env": {
    "DATABASE_URL": "$DATABASE_URL",
    "STORAGE_URL": "$STORAGE_URL",
    "BLOB_READ_WRITE_TOKEN": "$BLOB_READ_WRITE_TOKEN",
    "USE_TORCH": "ON"
  },
  "resources": {
    "cpu": "$CPU",
    "memory": "$MEMORY"
  }
}
EOF
)

echo -e "${YELLOW}Deploying container to Sakura Dok...${NC}"
echo ""

# Deploy the container
RESPONSE=$(curl -s -X POST "$API_ENDPOINT/containers" \
  -H "Authorization: Bearer $SAKURA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

# Check if deployment was successful
if echo "$RESPONSE" | grep -q "id"; then
    echo -e "${GREEN}✓ Deployment successful!${NC}"
    echo ""
    echo -e "${YELLOW}Container Details:${NC}"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    echo ""
    
    # Extract container ID if possible
    CONTAINER_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    if [ -n "$CONTAINER_ID" ]; then
        echo -e "${YELLOW}Useful commands:${NC}"
        echo -e "  Check status:  ${BLUE}curl -H 'Authorization: Bearer \$SAKURA_API_TOKEN' $API_ENDPOINT/containers/$CONTAINER_ID${NC}"
        echo -e "  View logs:     ${BLUE}curl -H 'Authorization: Bearer \$SAKURA_API_TOKEN' $API_ENDPOINT/containers/$CONTAINER_ID/logs${NC}"
        echo -e "  Stop container:${BLUE}curl -X DELETE -H 'Authorization: Bearer \$SAKURA_API_TOKEN' $API_ENDPOINT/containers/$CONTAINER_ID${NC}"
    fi
else
    echo -e "${RED}✗ Deployment failed${NC}"
    echo -e "${YELLOW}Response:${NC}"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
