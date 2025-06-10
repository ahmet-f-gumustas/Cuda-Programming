#!/bin/bash
set -e

# CUDA/GPU Video Pipeline Build Script
# Automated build system for the project

PROJECT_NAME="gpu_video_pipeline"
BUILD_DIR="build"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect platform
detect_platform() {
    if [ -f "/sys/firmware/devicetree/base/model" ]; then
        MODEL=$(cat /sys/firmware/devicetree/base/model)
        if [[ $MODEL == *"Jetson"* ]]; then
            echo "jetson"
            return
        fi
    fi
    echo "desktop"
}

# Function to check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing_deps=()
    
    # Check CMake
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    else
        CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
        print_status "CMake found: $CMAKE_VERSION"
    fi
    
    # Check NVCC
    if ! command_exists nvcc; then
        missing_deps+=("nvcc (CUDA Toolkit)")
    else
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_status "CUDA found: $CUDA_VERSION"
    fi
    
    # Check pkg-config
    if ! command_exists pkg-config; then
        missing_deps+=("pkg-config")
    fi
    
    # Check GStreamer
    if ! pkg-config --exists gstreamer-1.0; then
        missing_deps+=("gstreamer-1.0-dev")
    else
        GST_VERSION=$(pkg-config --modversion gstreamer-1.0)
        print_status "GStreamer found: $GST_VERSION"
    fi
    
    # Check OpenCV
    if ! pkg-config --exists opencv4; then
        missing_deps+=("libopencv-dev")
    else
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        print_status "OpenCV found: $OPENCV_VERSION"
    fi
    
    # Check Python dependencies
    if ! command_exists python3; then
        missing_deps+=("python3")
    else
        if ! python3 -c "import matplotlib, pandas, seaborn" 2>/dev/null; then
            missing_deps+=("python3-matplotlib python3-pandas python3-seaborn")
        else
            print_status "Python dependencies found"
        fi
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        print_status "Install dependencies with:"
        echo "  sudo apt update"
        echo "  sudo apt install ${missing_deps[*]}"
        exit 1
    fi
    
    print_status "All dependencies satisfied ✓"
}

# Function to setup project structure
setup_project_structure() {
    print_header "Setting Up Project Structure"
    
    local dirs=("include" "src" "scripts" "docs" "tests" "output/videos" "output/reports" "output/logs")
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$PROJECT_DIR/$dir" ]; then
            mkdir -p "$PROJECT_DIR/$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_status "Project structure ready ✓"
}

# Function to configure CMake
configure_cmake() {
    print_header "Configuring CMake"
    
    cd "$PROJECT_DIR"
    
    # Remove old build directory if exists
    if [ -d "$BUILD_DIR" ]; then
        print_warning "Removing existing build directory"
        rm -rf "$BUILD_DIR"
    fi
    
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Detect platform and set appropriate CUDA architectures
    PLATFORM=$(detect_platform)
    print_status "Platform detected: $PLATFORM"
    
    if [ "$PLATFORM" == "jetson" ]; then
        CUDA_ARCH="53;62;72;87"
        print_status "Using Jetson CUDA architectures: $CUDA_ARCH"
    else
        CUDA_ARCH="52;61;75;86;89"
        print_status "Using Desktop CUDA architectures: $CUDA_ARCH"
    fi
    
    # Configure with CMake
    cmake \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE:-Release} \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}" \
        ..
    
    print_status "CMake configuration complete ✓"
}

# Function to build project
build_project() {
    print_header "Building Project"
    
    cd "$PROJECT_DIR/$BUILD_DIR"
    
    # Determine number of cores for parallel build
    CORES=$(nproc)
    print_status "Building with $CORES parallel jobs"
    
    make -j"$CORES"
    
    if [ $? -eq 0 ]; then
        print_status "Build successful ✓"
        
        # Check if executable exists
        if [ -f "$PROJECT_NAME" ]; then
            print_status "Executable created: $BUILD_DIR/$PROJECT_NAME"
        else
            print_error "Executable not found after build"
            exit 1
        fi
    else
        print_error "Build failed"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    print_header "Running Tests"
    
    cd "$PROJECT_DIR/$BUILD_DIR"
    
    # Check if executable works
    if ./"$PROJECT_NAME" --help 2>/dev/null; then
        print_status "Basic executable test passed ✓"
    else
        print_warning "Executable test failed (might need runtime dependencies)"
    fi
    
    # Test CUDA functionality
    if nvidia-smi >/dev/null 2>&1; then
        print_status "NVIDIA GPU detected ✓"
    else
        print_warning "No NVIDIA GPU detected or nvidia-smi not available"
    fi
    
    # Test GStreamer plugins
    if gst-inspect-1.0 x264enc >/dev/null 2>&1; then
        print_status "x264enc plugin available ✓"
    else
        print_warning "x264enc plugin not found"
    fi
    
    print_status "Tests completed"
}

# Function to create test video
create_test_video() {
    print_header "Creating Test Video"
    
    cd "$PROJECT_DIR"
    
    local test_video="test_input_1080p60.h264"
    
    if [ -f "$test_video" ]; then
        print_status "Test video already exists: $test_video"
        return
    fi
    
    print_status "Creating 1080p60 test video..."
    
    gst-launch-1.0 videotestsrc num-buffers=1800 pattern=smpte \
        ! video/x-raw,width=1920,height=1080,framerate=60/1 \
        ! x264enc bitrate=8000 speed-preset=ultrafast \
        ! h264parse ! filesink location="$test_video" \
        2>/dev/null
    
    if [ -f "$test_video" ]; then
        print_status "Test video created: $test_video"
    else
        print_error "Failed to create test video"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -c, --clean          Clean build directory before building"
    echo "  -d, --debug          Build in debug mode"
    echo "  -r, --release        Build in release mode (default)"
    echo "  -t, --test           Run tests after building"
    echo "  -v, --create-video   Create test video"
    echo "  --install            Install to system after building"
    echo "  --deps-only          Only check dependencies and exit"
    echo ""
    echo "Environment Variables:"
    echo "  BUILD_TYPE           Set to 'Debug' or 'Release' (default: Release)"
    echo "  INSTALL_PREFIX       Installation prefix (default: /usr/local)"
    echo ""
    echo "Examples:"
    echo "  $0                   # Build in release mode"
    echo "  $0 -d -t             # Debug build with tests"
    echo "  $0 -c -r --install   # Clean, release build and install"
}

# Main function
main() {
    local clean_build=false
    local run_tests_flag=false
    local create_video_flag=false
    local install_flag=false
    local deps_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--clean)
                clean_build=true
                shift
                ;;
            -d|--debug)
                export BUILD_TYPE="Debug"
                shift
                ;;
            -r|--release)
                export BUILD_TYPE="Release"
                shift
                ;;
            -t|--test)
                run_tests_flag=true
                shift
                ;;
            -v|--create-video)
                create_video_flag=true
                shift
                ;;
            --install)
                install_flag=true
                shift
                ;;
            --deps-only)
                deps_only=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_header "CUDA/GPU Video Pipeline Build System"
    print_status "Build Type: ${BUILD_TYPE:-Release}"
    print_status "Platform: $(detect_platform)"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    if [ "$deps_only" = true ]; then
        print_status "Dependencies check completed"
        exit 0
    fi
    
    # Setup project structure
    setup_project_structure
    
    # Clean if requested
    if [ "$clean_build" = true ]; then
        print_status "Cleaning build directory"
        rm -rf "$PROJECT_DIR/$BUILD_DIR"
    fi
    
    # Configure and build
    configure_cmake
    build_project
    
    # Run tests if requested
    if [ "$run_tests_flag" = true ]; then
        run_tests
    fi
    
    # Create test video if requested
    if [ "$create_video_flag" = true ]; then
        create_test_video
    fi
    
    # Install if requested
    if [ "$install_flag" = true ]; then
        print_header "Installing"
        cd "$PROJECT_DIR/$BUILD_DIR"
        sudo make install
        print_status "Installation complete ✓"
    fi
    
    print_header "Build Complete"
    print_status "Executable: $BUILD_DIR/$PROJECT_NAME"
    print_status "Run with: ./$BUILD_DIR/$PROJECT_NAME"
    
    if [ "$create_video_flag" = true ]; then
        print_status "Test video: test_input_1080p60.h264"
    fi
    
    echo ""
    print_status "To run the pipeline:"
    echo "  cd $PROJECT_DIR"
    echo "  ./$BUILD_DIR/$PROJECT_NAME"
    echo ""
    print_status "Happy coding! 🚀"
}

# Run main function with all arguments
main "$@"