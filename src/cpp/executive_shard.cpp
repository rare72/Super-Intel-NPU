#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <string>
#include <level_zero/ze_api.h> // Intel Level Zero Header
#include <openvino/openvino.hpp>

// Configuration Constants
const char* SHM_NAME = "/offering_tensor_shm";
const char* REPORT_PIPE_PATH = "/tmp/offering_report";
const char* COMMAND_PIPE_PATH = "/tmp/offering_command";
const size_t TENSOR_SIZE = 4096 * sizeof(float); // Example 1x4096 float32

// Helper function to write to the named pipe
void report_status(const std::string& msg) {
    int fd = open(REPORT_PIPE_PATH, O_WRONLY | O_NONBLOCK);
    if (fd != -1) {
        std::string report = "STATUS:" + msg + "\n";
        write(fd, report.c_str(), report.length());
        close(fd);
    }
}

// Helper function to log messages
void log_message(const std::string& msg) {
    std::cout << "[Executive] " << msg << std::endl;
}

// Hardware Discovery using Level Zero
void discover_hardware() {
    log_message("Initializing Level Zero...");
    ze_result_t result = zeInit(ZE_INIT_FLAG_GPU_ONLY | ZE_INIT_FLAG_VPU_ONLY);
    if (result != ZE_RESULT_SUCCESS) {
        log_message("Level Zero initialization failed (or no devices found). Continuing with CPU fallback possibility.");
        return;
    }

    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, nullptr);
    std::vector<ze_driver_handle_t> drivers(driverCount);
    zeDriverGet(&driverCount, drivers.data());

    for (auto driver : drivers) {
        uint32_t deviceCount = 0;
        zeDeviceGet(driver, &deviceCount, nullptr);
        std::vector<ze_device_handle_t> devices(deviceCount);
        zeDeviceGet(driver, &deviceCount, devices.data());

        for (auto device : devices) {
            ze_device_properties_t props = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            zeDeviceGetProperties(device, &props);

            if (props.type == ZE_DEVICE_TYPE_GPU) {
                log_message("Found Intel GPU: " + std::string(props.name));
            } else if (props.type == ZE_DEVICE_TYPE_VPU) { // VPU is the NPU
                log_message("Found Intel NPU: " + std::string(props.name));
            }
        }
    }
}

int main() {
    log_message("Starting Executive Shard...");

    // 1. Hardware Discovery
    discover_hardware();

    // 2. Setup POSIX Shared Memory
    log_message("Setting up Shared Memory...");
    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory" << std::endl;
        return 1;
    }
    ftruncate(shm_fd, TENSOR_SIZE);
    float* shared_ptr = (float*)mmap(0, TENSOR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_ptr == MAP_FAILED) {
         std::cerr << "Failed to mmap shared memory" << std::endl;
         return 1;
    }

    // 3. Setup Pipes (Create if not exist)
    mkfifo(REPORT_PIPE_PATH, 0666);
    mkfifo(COMMAND_PIPE_PATH, 0666);

    log_message("Resources ready. Signaling READY.");
    report_status("READY");

    // 4. Main Executive Loop
    // In a real implementation, we would listen to COMMAND_PIPE_PATH.
    // Here we simulate the waiting and processing loop.

    // Open command pipe for reading
    int cmd_fd = open(COMMAND_PIPE_PATH, O_RDONLY | O_NONBLOCK);

    bool running = true;
    char buffer[1024];

    while (running) {
        // Simple polling for commands (in prod use select/epoll)
        if (cmd_fd != -1) {
            ssize_t bytes = read(cmd_fd, buffer, sizeof(buffer)-1);
            if (bytes > 0) {
                buffer[bytes] = '\0';
                std::string cmd(buffer);
                if (cmd.find("EXIT") != std::string::npos) {
                    running = false;
                } else if (cmd.find("PROCESS") != std::string::npos) {
                    // Simulate NPU Inference
                    // log_message("Processing on NPU...");

                    // Simulate writing tensor data (dummy values)
                    // In real code: memcpy(shared_ptr, npu_output, TENSOR_SIZE);
                    shared_ptr[0] = 0.123f;
                    shared_ptr[1] = 0.456f;

                    // Signal data is ready
                    report_status("DATA_READY");
                }
            }
        }

        // Sleep to prevent CPU spin
        usleep(10000); // 10ms

        // Re-open pipe if it was closed or not ready initially
        if (cmd_fd == -1) {
             cmd_fd = open(COMMAND_PIPE_PATH, O_RDONLY | O_NONBLOCK);
        }
    }

    // Cleanup
    log_message("Shutting down.");
    if (cmd_fd != -1) close(cmd_fd);
    munmap(shared_ptr, TENSOR_SIZE);
    shm_unlink(SHM_NAME);

    return 0;
}
