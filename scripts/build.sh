#!/bin/bash

# Video Diffusion Benchmark Suite - Build Script
# This script handles building Docker images and setting up the development environment

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_LOG="$PROJECT_ROOT/build.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build configuration
DEFAULT_TARGET="development"
DEFAULT_REGISTRY="ghcr.io/danieleschmidt/vid-diffusion-benchmark"
DOCKER_BUILDKIT=1

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$BUILD_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$BUILD_LOG"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$BUILD_LOG"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$BUILD_LOG"
}

# Help function
show_help() {
    cat << EOF
Video Diffusion Benchmark Suite - Build Script

USAGE:
    $0 [OPTIONS] [COMMAND]

COMMANDS:
    all                 Build all images (default)
    base                Build base images only
    services            Build service images only
    models              Build model images only
    clean               Clean build artifacts and images
    test                Run build tests
    push                Push images to registry
    pull                Pull images from registry

OPTIONS:
    -t, --target TARGET     Build target (development|production) [default: development]
    -r, --registry REGISTRY Registry prefix [default: $DEFAULT_REGISTRY]
    -v, --version VERSION   Image version tag [default: latest]
    -j, --jobs JOBS         Number of parallel build jobs [default: 2]
    -p, --platform PLATFORM Target platform [default: linux/amd64]
    -f, --force             Force rebuild without cache
    -q, --quiet             Suppress build output
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Build all images for development
    $0 --target production  # Build production images
    $0 models               # Build only model images
    $0 clean                # Clean all build artifacts
    $0 push --version 1.0.0 # Push images with specific version

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY         Override default registry
    BUILD_TARGET            Override default build target
    BUILD_VERSION           Override default version
    BUILD_PARALLEL_JOBS     Override default parallel jobs

EOF
}

# Parse command line arguments
parse_args() {
    TARGET="$DEFAULT_TARGET"
    REGISTRY="${DOCKER_REGISTRY:-$DEFAULT_REGISTRY}"
    VERSION="${BUILD_VERSION:-latest}"
    JOBS="${BUILD_PARALLEL_JOBS:-2}"
    PLATFORM="linux/amd64"
    FORCE_REBUILD=false
    QUIET=false
    COMMAND="all"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -j|--jobs)
                JOBS="$2"
                shift 2
                ;;
            -p|--platform)
                PLATFORM="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_REBUILD=true
                shift
                ;;
            -q|--quiet)
                QUIET=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            all|base|services|models|clean|test|push|pull)
                COMMAND="$1"
                shift
                ;;
            *)
                error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate arguments
    if [[ ! "$TARGET" =~ ^(development|production)$ ]]; then
        error "Invalid target: $TARGET. Must be 'development' or 'production'"
        exit 1
    fi
    
    if [[ ! "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
        error "Invalid jobs count: $JOBS. Must be a positive integer"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        warn "Docker Compose not found. Some features may not work"
    fi
    
    # Check BuildKit
    if [[ "$DOCKER_BUILDKIT" == "1" ]]; then
        if ! docker buildx version &> /dev/null; then
            warn "Docker BuildKit not available. Falling back to legacy build"
            DOCKER_BUILDKIT=0
        fi
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ "$available_space" -lt 10485760 ]]; then # 10GB in KB
        warn "Low disk space. Build may fail if space runs out"
    fi
    
    log "Prerequisites check completed"
}

# Build Docker image
build_image() {
    local name="$1"
    local dockerfile="$2"
    local context="$3"
    local build_args="$4"
    
    local image_tag="$REGISTRY/$name:$VERSION"
    local cache_args=""
    local output_args=""
    
    if [[ "$QUIET" == "true" ]]; then
        output_args="--quiet"
    fi
    
    if [[ "$FORCE_REBUILD" == "false" ]]; then
        cache_args="--cache-from $image_tag"
    else
        cache_args="--no-cache"
    fi
    
    info "Building image: $image_tag"
    
    if [[ "$DOCKER_BUILDKIT" == "1" ]]; then
        DOCKER_BUILDKIT=1 docker build \
            --platform "$PLATFORM" \
            --target "$TARGET" \
            --tag "$image_tag" \
            --file "$dockerfile" \
            $cache_args \
            $build_args \
            $output_args \
            "$context" >> "$BUILD_LOG" 2>&1
    else
        docker build \
            --target "$TARGET" \
            --tag "$image_tag" \
            --file "$dockerfile" \
            $cache_args \
            $build_args \
            $output_args \
            "$context" >> "$BUILD_LOG" 2>&1
    fi
    
    if [[ $? -eq 0 ]]; then
        log "Successfully built: $image_tag"
    else
        error "Failed to build: $image_tag"
        return 1
    fi
}

# Build base images
build_base_images() {
    log "Building base images..."
    
    local base_dir="$PROJECT_ROOT/docker/base"
    
    if [[ -d "$base_dir" ]]; then
        build_image "pytorch-base" "$base_dir/Dockerfile.pytorch" "$base_dir" ""
        build_image "cuda-base" "$base_dir/Dockerfile.cuda" "$base_dir" ""
        build_image "runtime-base" "$base_dir/Dockerfile.runtime" "$base_dir" ""
    else
        warn "Base images directory not found, skipping base images"
    fi
}

# Build service images
build_service_images() {
    log "Building service images..."
    
    # Main benchmark service
    build_image "benchmark" "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT" "--build-arg TARGET=$TARGET"
    
    # API service
    local api_dir="$PROJECT_ROOT/docker/services/api"
    if [[ -d "$api_dir" ]]; then
        build_image "api" "$api_dir/Dockerfile" "$api_dir" ""
    fi
    
    # Dashboard service
    local dashboard_dir="$PROJECT_ROOT/docker/services/dashboard"
    if [[ -d "$dashboard_dir" ]]; then
        build_image "dashboard" "$dashboard_dir/Dockerfile" "$dashboard_dir" ""
    fi
    
    # Worker service
    local worker_dir="$PROJECT_ROOT/docker/services/worker"
    if [[ -d "$worker_dir" ]]; then
        build_image "worker" "$worker_dir/Dockerfile" "$worker_dir" ""
    fi
}

# Build model images
build_model_images() {
    log "Building model images..."
    
    local models_dir="$PROJECT_ROOT/docker/models"
    
    if [[ ! -d "$models_dir" ]]; then
        warn "Models directory not found, skipping model images"
        return
    fi
    
    # Build model images in parallel
    local pids=()
    local active_jobs=0
    
    for model_dir in "$models_dir"/*; do
        if [[ -d "$model_dir" && -f "$model_dir/Dockerfile" ]]; then
            local model_name=$(basename "$model_dir")
            
            # Skip template directory
            if [[ "$model_name" == "template" ]]; then
                continue
            fi
            
            # Wait if we've reached the maximum number of parallel jobs
            while [[ "$active_jobs" -ge "$JOBS" ]]; do
                wait -n
                active_jobs=$((active_jobs - 1))
            done
            
            # Build model image in background
            (
                build_image "$model_name" "$model_dir/Dockerfile" "$model_dir" ""
            ) &
            
            pids+=($!)
            active_jobs=$((active_jobs + 1))
        fi
    done
    
    # Wait for all background jobs to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
}

# Clean build artifacts
clean_build() {
    log "Cleaning build artifacts..."
    
    # Remove dangling images
    docker image prune -f >> "$BUILD_LOG" 2>&1 || true
    
    # Remove build cache
    if [[ "$DOCKER_BUILDKIT" == "1" ]]; then
        docker builder prune -f >> "$BUILD_LOG" 2>&1 || true
    fi
    
    # Remove project-specific images if requested
    if [[ "${CLEAN_ALL:-false}" == "true" ]]; then
        docker images "$REGISTRY/*" -q | xargs -r docker rmi -f >> "$BUILD_LOG" 2>&1 || true
    fi
    
    # Clean build logs older than 7 days
    find "$PROJECT_ROOT" -name "build.log*" -mtime +7 -delete 2>/dev/null || true
    
    log "Clean completed"
}

# Test built images
test_build() {
    log "Testing built images..."
    
    local test_passed=0
    local test_failed=0
    
    # Test main benchmark image
    local benchmark_image="$REGISTRY/benchmark:$VERSION"
    if docker images "$benchmark_image" -q &> /dev/null; then
        info "Testing benchmark image..."
        if docker run --rm "$benchmark_image" python -c "import vid_diffusion_bench; print('OK')" &> /dev/null; then
            log "Benchmark image test: PASSED"
            test_passed=$((test_passed + 1))
        else
            error "Benchmark image test: FAILED"
            test_failed=$((test_failed + 1))
        fi
    fi
    
    # Test model images
    for model_dir in "$PROJECT_ROOT/docker/models"/*; do
        if [[ -d "$model_dir" && -f "$model_dir/Dockerfile" ]]; then
            local model_name=$(basename "$model_dir")
            if [[ "$model_name" == "template" ]]; then
                continue
            fi
            
            local model_image="$REGISTRY/$model_name:$VERSION"
            if docker images "$model_image" -q &> /dev/null; then
                info "Testing $model_name image..."
                if docker run --rm "$model_image" python -c "print('Health check OK')" &> /dev/null; then
                    log "$model_name image test: PASSED"
                    test_passed=$((test_passed + 1))
                else
                    error "$model_name image test: FAILED"
                    test_failed=$((test_failed + 1))
                fi
            fi
        fi
    done
    
    log "Test summary: $test_passed passed, $test_failed failed"
    return $test_failed
}

# Push images to registry
push_images() {
    log "Pushing images to registry: $REGISTRY"
    
    # Get list of built images
    local images=($(docker images "$REGISTRY/*:$VERSION" --format "{{.Repository}}:{{.Tag}}"))
    
    if [[ ${#images[@]} -eq 0 ]]; then
        warn "No images found to push"
        return
    fi
    
    # Push images
    for image in "${images[@]}"; do
        info "Pushing $image..."
        docker push "$image" >> "$BUILD_LOG" 2>&1
        if [[ $? -eq 0 ]]; then
            log "Successfully pushed: $image"
        else
            error "Failed to push: $image"
        fi
    done
}

# Pull images from registry
pull_images() {
    log "Pulling images from registry: $REGISTRY"
    
    # List of expected images
    local images=(
        "benchmark"
        "api"
        "dashboard"
        "worker"
    )
    
    # Add model images
    for model_dir in "$PROJECT_ROOT/docker/models"/*; do
        if [[ -d "$model_dir" && -f "$model_dir/Dockerfile" ]]; then
            local model_name=$(basename "$model_dir")
            if [[ "$model_name" != "template" ]]; then
                images+=("$model_name")
            fi
        fi
    done
    
    # Pull images
    for image_name in "${images[@]}"; do
        local image="$REGISTRY/$image_name:$VERSION"
        info "Pulling $image..."
        docker pull "$image" >> "$BUILD_LOG" 2>&1 || warn "Failed to pull: $image"
    done
}

# Main execution
main() {
    parse_args "$@"
    
    log "Starting build process..."
    log "Target: $TARGET"
    log "Registry: $REGISTRY"
    log "Version: $VERSION"
    log "Platform: $PLATFORM"
    log "Jobs: $JOBS"
    log "Command: $COMMAND"
    
    check_prerequisites
    
    case "$COMMAND" in
        all)
            build_base_images
            build_service_images
            build_model_images
            ;;
        base)
            build_base_images
            ;;
        services)
            build_service_images
            ;;
        models)
            build_model_images
            ;;
        clean)
            clean_build
            ;;
        test)
            test_build
            exit $?
            ;;
        push)
            push_images
            ;;
        pull)
            pull_images
            ;;
        *)
            error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
    
    log "Build process completed successfully"
}

# Run main function
main "$@"