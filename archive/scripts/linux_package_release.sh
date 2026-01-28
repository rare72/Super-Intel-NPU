#!/bin/bash
set -e

echo ">>> Starting Release Packaging..."

BUILD_DIR="src/cpp/build"
RELEASE_DIR="release_package"

# 1. Compile C++ Executive
echo ">>> Compiling C++ Executive..."
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Check if cmake is available
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake could not be found. Please run setup_env.sh first."
    exit 1
fi

# Attempt cmake configuration
# Note: In a sandbox environment without Intel drivers, this might fail or warn.
# We proceed if make fails only to allow packaging scripts for the user.
echo "Running CMake..."
if cmake ..; then
    echo "Running Make..."
    make -j$(nproc)
else
    echo ">>> Warning: CMake configuration failed (likely due to missing Intel drivers in this environment)."
    echo ">>> Skipping compilation step. The user must compile on the target machine."
    # Create a dummy binary for directory structure if needed, or just warn
fi

cd ../../../

# 2. Create Release Directory
echo ">>> Creating Release Structure at $RELEASE_DIR..."
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR/bin
mkdir -p $RELEASE_DIR/models
mkdir -p $RELEASE_DIR/scripts
mkdir -p $RELEASE_DIR/documentation

# 3. Copy Artifacts
echo ">>> Copying Files..."

# Python Source
cp src/python/supervisor.py $RELEASE_DIR/
cp src/python/bake_model.py $RELEASE_DIR/
cp src/python/nncf_config.json $RELEASE_DIR/

# C++ Binary (if exists)
if [ -f "$BUILD_DIR/executive_shard" ]; then
    cp $BUILD_DIR/executive_shard $RELEASE_DIR/bin/
else
    echo ">>> Warning: 'executive_shard' binary not found. User needs to compile it."
fi

# Scripts
cp scripts/setup_env.sh $RELEASE_DIR/scripts/
cp scripts/check_intel_hw.py $RELEASE_DIR/scripts/

# Documentation
cp README.md $RELEASE_DIR/
if [ -d "documentation" ]; then
    cp -r documentation/* $RELEASE_DIR/documentation/
fi

# 4. Create Run Script
echo ">>> Creating Launch Script..."
cat << 'EOF' > $RELEASE_DIR/run_offering.sh
#!/bin/bash
# Wrapper to run the offering
source scripts/setup_env.sh # Activate env (simplified logic)
python3 supervisor.py
EOF
chmod +x $RELEASE_DIR/run_offering.sh

echo ">>> Release Package Created Successfully at $RELEASE_DIR"
