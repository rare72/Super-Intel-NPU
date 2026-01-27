import os
import sys
import argparse
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BakeQwenInt8")

def bake_int8_dynamic(staging_dir, output_dir):
    logger.info(f"Starting Qwen3 INT8 Dynamic Bake...")
    logger.info(f"Source: {staging_dir}")
    logger.info(f"Target: {output_dir}")

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct optimum-cli command
    # optimum-cli export openvino --model ... --task text-generation-with-past --weight-format int8 --trust-remote-code ...
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", staging_dir,
        "--task", "text-generation-with-past",
        "--weight-format", "int8",
        "--trust-remote-code",
        output_dir
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    try:
        # Run the CLI command
        result = subprocess.run(cmd, check=True, text=True)
        logger.info("Bake process completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Bake failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("optimum-cli not found. Please ensure 'optimum-intel[openvino]' is installed.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging_dir", default="./model_staging_qwen")
    parser.add_argument("--output_dir", default="./models/pv1-qwen3_int8")
    args = parser.parse_args()

    bake_int8_dynamic(args.staging_dir, args.output_dir)
