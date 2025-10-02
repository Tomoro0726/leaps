#!/bin/bash
# Build script for Sakura Internet High-Power Dok

set -e

# Configuration
IMAGE_NAME="leaps-worker"
VERSION="latest"
DOCKERFILE="Dockerfile.sakura"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LEAPS - Sakura Dok Build Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Parse command line arguments
REGISTRY=""
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -r, --registry REGISTRY  Container registry URL"
            echo "  -p, --push               Push image to registry after build"
            echo "  -v, --version VERSION    Image version tag (default: latest)"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 -r myregistry.example.com/myproject -p -v v1.0.0"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Set image tag
if [ -n "$REGISTRY" ]; then
    IMAGE_TAG="$REGISTRY/$IMAGE_NAME:$VERSION"
else
    IMAGE_TAG="$IMAGE_NAME:$VERSION"
fi

echo -e "${YELLOW}Building Docker image...${NC}"
echo -e "Image tag: ${GREEN}$IMAGE_TAG${NC}"
echo -e "Dockerfile: ${GREEN}$DOCKERFILE${NC}"
echo ""

# Build the image
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    
    # Show image info
    echo -e "${YELLOW}Image information:${NC}"
    docker images "$IMAGE_TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    echo ""
    
    # Push if requested
    if [ "$PUSH" = true ]; then
        if [ -z "$REGISTRY" ]; then
            echo -e "${RED}Error: Registry not specified. Use -r option to specify registry.${NC}"
            exit 1
        fi
        
        echo -e "${YELLOW}Pushing image to registry...${NC}"
        docker push "$IMAGE_TAG"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Push successful!${NC}"
        else
            echo -e "${RED}✗ Push failed${NC}"
            exit 1
        fi
    fi
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Build Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "To run locally:"
    echo -e "  ${YELLOW}docker run --gpus all -e DATABASE_URL=... -e STORAGE_URL=... -e BLOB_READ_WRITE_TOKEN=... $IMAGE_TAG${NC}"
    echo ""
    echo -e "To push to registry (if not already pushed):"
    echo -e "  ${YELLOW}docker push $IMAGE_TAG${NC}"
    echo ""
    
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
