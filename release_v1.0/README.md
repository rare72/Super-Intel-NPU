# New Offering: High-Performance NPU Inference

## Overview
This package contains the optimized "New Offering" inference engine for Intel Core Ultra NPU.
It utilizes a hybrid Python/C++ architecture with Zero-Copy Shared Memory handoffs.

## Requirements
- Intel Core Ultra Processor (Series 1 or 2)
- Linux Kernel 6.8+ (for NPU driver support)
- Python 3.10+
- Intel Level Zero Drivers

## Installation
1. Ensure you have the Intel NPU drivers installed:
   `sudo apt install intel-level-zero-npu intel-level-zero-gpu`

2. Install Python dependencies:
   `pip install numpy torch openvino optimum-intel transformers`

## Usage
To run the interactive chat:
`./run.sh`

To run a single prompt:
`./run.sh --prompt "Why is the sky blue?"`

## Troubleshooting
- If you see **ZE_RESULT_ERROR_UNKNOWN**, ensure no other NPU processes are running.
- Use `./run.sh --device CPU` to fallback if the NPU is unstable.
