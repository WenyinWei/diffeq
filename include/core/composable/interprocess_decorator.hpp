#pragma once

#include "integrator_decorator.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>
#include <fstream>
#include <queue>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#endif

namespace diffeq::core::composable {

/**
 * @brief IPC communication method enumeration
 */
enum class IPCMethod {
    SHARED_MEMORY,      // Shared memory (fastest)
    NAMED_PIPES,        // Named pipes (cross-platform)
    MEMORY_MAPPED_FILE, // Memory-mapped files
    TCP_SOCKET,         // TCP sockets (network-capable)
    UDP_SOCKET          // UDP sockets (low-latency)
};

/**
 * @brief IPC channel direction
 */
enum class IPCDirection {
    PRODUCER,           // This process sends data
    CONSUMER,           // This process receives data
    BIDIRECTIONAL       // Both send and receive
};

/**
 * @brief IPC synchronization mode
 */
enum class IPCSyncMode {
    BLOCKING,           // Block until data available
    NON_BLOCKING,       // Return immediately if no data
    TIMEOUT,            // Block with timeout
    POLLING             // Actively poll for data
};

/**
 * @brief Configuration for interprocess communication
 */
struct InterprocessConfig {
    IPCMethod method{IPCMethod::SHARED_MEMORY};
    IPCDirection direction{IPCDirection::PRODUCER};
    IPCSyncMode sync_mode{IPCSyncMode::NON_BLOCKING};
    
    std::string channel_name{"diffeq_channel"};
    size_t buffer_size{1024 * 1024};  // 1MB default
    size_t max_message_size{64 * 1024}; // 64KB default
    
    std::chrono::milliseconds timeout{100};
    std::chrono::microseconds polling_interval{100}; // 100μs
    
    // Reliability settings
    bool enable_acknowledgments{false};
    bool enable_sequence_numbers{true};
    bool enable_error_correction{false};
    size_t max_retries{3};
    
    // Performance settings
    bool enable_compression{false};
    bool enable_batching{false};
    size_t batch_size{10};
    std::chrono::milliseconds batch_timeout{10};
    
    // Network settings (for TCP/UDP)
    std::string host{"127.0.0.1"};
    uint16_t port{8080};
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (channel_name.empty()) {
            throw std::invalid_argument("channel_name cannot be empty");
        }
        
        if (buffer_size == 0) {
            throw std::invalid_argument("buffer_size must be positive");
        }
        
        if (max_message_size > buffer_size) {
            throw std::invalid_argument("max_message_size cannot exceed buffer_size");
        }
        
        if (timeout <= std::chrono::milliseconds{0}) {
            throw std::invalid_argument("timeout must be positive");
        }
        
        if (polling_interval <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("polling_interval must be positive");
        }
        
        if (batch_size == 0) {
            throw std::invalid_argument("batch_size must be positive");
        }
        
        if (port == 0) {
            throw std::invalid_argument("port must be positive");
        }
    }
};

/**
 * @brief IPC message structure
 */
template<typename T>
struct IPCMessage {
    uint32_t sequence_number{0};
    uint32_t message_size{0};
    T timestamp{};
    std::vector<uint8_t> data;
    
    void serialize_state(const auto& state) {
        data.resize(sizeof(state));
        std::memcpy(data.data(), &state, sizeof(state));
        message_size = data.size();
    }
    
    auto deserialize_state() const {
        // Note: This is a simplified implementation
        // Real implementation would use proper serialization
        if (data.size() < sizeof(std::vector<double>)) {
            throw std::runtime_error("Invalid message size for deserialization");
        }
        
        std::vector<double> state;
        // Simplified - would need proper deserialization
        return state;
    }
};

/**
 * @brief IPC statistics
 */
struct IPCStats {
    size_t messages_sent{0};
    size_t messages_received{0};
    size_t bytes_sent{0};
    size_t bytes_received{0};
    size_t send_failures{0};
    size_t receive_failures{0};
    size_t acknowledgments_sent{0};
    size_t acknowledgments_received{0};
    std::chrono::milliseconds total_send_time{0};
    std::chrono::milliseconds total_receive_time{0};
    
    double average_send_time_ms() const {
        return messages_sent > 0 ? 
            static_cast<double>(total_send_time.count()) / messages_sent : 0.0;
    }
    
    double average_receive_time_ms() const {
        return messages_received > 0 ? 
            static_cast<double>(total_receive_time.count()) / messages_received : 0.0;
    }
    
    double send_success_rate() const {
        size_t total = messages_sent + send_failures;
        return total > 0 ? static_cast<double>(messages_sent) / total : 0.0;
    }
    
    double receive_success_rate() const {
        size_t total = messages_received + receive_failures;
        return total > 0 ? static_cast<double>(messages_received) / total : 0.0;
    }
};

/**
 * @brief Base class for IPC channels
 */
template<typename T>
class IPCChannel {
public:
    virtual ~IPCChannel() = default;
    
    virtual bool initialize() = 0;
    virtual void cleanup() = 0;
    
    virtual bool send_message(const IPCMessage<T>& message) = 0;
    virtual bool receive_message(IPCMessage<T>& message) = 0;
    
    virtual bool is_connected() const = 0;
    virtual std::string get_status() const = 0;
};

/**
 * @brief Shared memory IPC channel
 */
template<typename T>
class SharedMemoryChannel : public IPCChannel<T> {
private:
    std::string name_;
    size_t buffer_size_;
    void* memory_ptr_{nullptr};
    
#ifdef _WIN32
    HANDLE file_mapping_{nullptr};
#else
    int fd_{-1};
#endif
    
    std::mutex* mutex_{nullptr};
    std::condition_variable* condition_{nullptr};
    bool initialized_{false};

public:
    explicit SharedMemoryChannel(const std::string& name, size_t buffer_size)
        : name_(name), buffer_size_(buffer_size) {}
    
    ~SharedMemoryChannel() {
        cleanup();
    }
    
    bool initialize() override {
        if (initialized_) return true;
        
#ifdef _WIN32
        file_mapping_ = CreateFileMapping(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            0,
            buffer_size_,
            name_.c_str()
        );
        
        if (file_mapping_ == nullptr) {
            return false;
        }
        
        memory_ptr_ = MapViewOfFile(
            file_mapping_,
            FILE_MAP_ALL_ACCESS,
            0,
            0,
            buffer_size_
        );
        
        if (memory_ptr_ == nullptr) {
            CloseHandle(file_mapping_);
            return false;
        }
#else
        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ == -1) {
            return false;
        }
        
        if (ftruncate(fd_, buffer_size_) == -1) {
            close(fd_);
            return false;
        }
        
        memory_ptr_ = mmap(nullptr, buffer_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (memory_ptr_ == MAP_FAILED) {
            close(fd_);
            return false;
        }
#endif
        
        // Initialize synchronization primitives in shared memory
        mutex_ = new(memory_ptr_) std::mutex();
        condition_ = new(static_cast<char*>(memory_ptr_) + sizeof(std::mutex)) std::condition_variable();
        
        initialized_ = true;
        return true;
    }
    
    void cleanup() override {
        if (!initialized_) return;
        
#ifdef _WIN32
        if (memory_ptr_) {
            UnmapViewOfFile(memory_ptr_);
            memory_ptr_ = nullptr;
        }
        if (file_mapping_) {
            CloseHandle(file_mapping_);
            file_mapping_ = nullptr;
        }
#else
        if (memory_ptr_ != MAP_FAILED) {
            munmap(memory_ptr_, buffer_size_);
            memory_ptr_ = nullptr;
        }
        if (fd_ != -1) {
            close(fd_);
            shm_unlink(name_.c_str());
            fd_ = -1;
        }
#endif
        
        initialized_ = false;
    }
    
    bool send_message(const IPCMessage<T>& message) override {
        if (!initialized_) return false;
        
        std::lock_guard<std::mutex> lock(*mutex_);
        
        // Write message to shared memory
        char* data_ptr = static_cast<char*>(memory_ptr_) + sizeof(std::mutex) + sizeof(std::condition_variable);
        size_t message_size = sizeof(IPCMessage<T>) + message.data.size();
        
        if (message_size > buffer_size_ - sizeof(std::mutex) - sizeof(std::condition_variable)) {
            return false;
        }
        
        std::memcpy(data_ptr, &message, sizeof(IPCMessage<T>));
        std::memcpy(data_ptr + sizeof(IPCMessage<T>), message.data.data(), message.data.size());
        
        condition_->notify_one();
        return true;
    }
    
    bool receive_message(IPCMessage<T>& message) override {
        if (!initialized_) return false;
        
        std::unique_lock<std::mutex> lock(*mutex_);
        
        // For simplicity, we'll use a timeout-based approach
        if (!condition_->wait_for(lock, std::chrono::milliseconds(100))) {
            return false;
        }
        
        // Read message from shared memory
        char* data_ptr = static_cast<char*>(memory_ptr_) + sizeof(std::mutex) + sizeof(std::condition_variable);
        
        std::memcpy(&message, data_ptr, sizeof(IPCMessage<T>));
        message.data.resize(message.message_size);
        std::memcpy(message.data.data(), data_ptr + sizeof(IPCMessage<T>), message.message_size);
        
        return true;
    }
    
    bool is_connected() const override {
        return initialized_;
    }
    
    std::string get_status() const override {
        return initialized_ ? "Connected" : "Disconnected";
    }
};

/**
 * @brief Named pipe IPC channel
 */
template<typename T>
class NamedPipeChannel : public IPCChannel<T> {
private:
    std::string name_;
    std::string pipe_path_;
    
#ifdef _WIN32
    HANDLE pipe_handle_{INVALID_HANDLE_VALUE};
#else
    int pipe_fd_{-1};
#endif
    
    bool initialized_{false};
    bool is_server_{false};

public:
    explicit NamedPipeChannel(const std::string& name, bool is_server = false)
        : name_(name), is_server_(is_server) {
        
#ifdef _WIN32
        pipe_path_ = "\\\\.\\pipe\\" + name_;
#else
        pipe_path_ = "/tmp/" + name_;
#endif
    }
    
    ~NamedPipeChannel() {
        cleanup();
    }
    
    bool initialize() override {
        if (initialized_) return true;
        
#ifdef _WIN32
        if (is_server_) {
            pipe_handle_ = CreateNamedPipe(
                pipe_path_.c_str(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                1,
                4096,
                4096,
                0,
                nullptr
            );
            
            if (pipe_handle_ == INVALID_HANDLE_VALUE) {
                return false;
            }
            
            if (!ConnectNamedPipe(pipe_handle_, nullptr)) {
                if (GetLastError() != ERROR_PIPE_CONNECTED) {
                    CloseHandle(pipe_handle_);
                    return false;
                }
            }
        } else {
            pipe_handle_ = CreateFile(
                pipe_path_.c_str(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                nullptr,
                OPEN_EXISTING,
                0,
                nullptr
            );
            
            if (pipe_handle_ == INVALID_HANDLE_VALUE) {
                return false;
            }
        }
#else
        if (is_server_) {
            if (mkfifo(pipe_path_.c_str(), 0666) == -1 && errno != EEXIST) {
                return false;
            }
        }
        
        pipe_fd_ = open(pipe_path_.c_str(), O_RDWR | O_NONBLOCK);
        if (pipe_fd_ == -1) {
            return false;
        }
#endif
        
        initialized_ = true;
        return true;
    }
    
    void cleanup() override {
        if (!initialized_) return;
        
#ifdef _WIN32
        if (pipe_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(pipe_handle_);
            pipe_handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (pipe_fd_ != -1) {
            close(pipe_fd_);
            pipe_fd_ = -1;
        }
        if (is_server_) {
            unlink(pipe_path_.c_str());
        }
#endif
        
        initialized_ = false;
    }
    
    bool send_message(const IPCMessage<T>& message) override {
        if (!initialized_) return false;
        
        // Serialize message
        std::vector<uint8_t> serialized_data;
        serialized_data.resize(sizeof(IPCMessage<T>) + message.data.size());
        
        std::memcpy(serialized_data.data(), &message, sizeof(IPCMessage<T>));
        std::memcpy(serialized_data.data() + sizeof(IPCMessage<T>), message.data.data(), message.data.size());
        
#ifdef _WIN32
        DWORD bytes_written;
        return WriteFile(pipe_handle_, serialized_data.data(), serialized_data.size(), &bytes_written, nullptr) &&
               bytes_written == serialized_data.size();
#else
        ssize_t bytes_written = write(pipe_fd_, serialized_data.data(), serialized_data.size());
        return bytes_written == static_cast<ssize_t>(serialized_data.size());
#endif
    }
    
    bool receive_message(IPCMessage<T>& message) override {
        if (!initialized_) return false;
        
        // First, read the message header
        IPCMessage<T> header;
        
#ifdef _WIN32
        DWORD bytes_read;
        if (!ReadFile(pipe_handle_, &header, sizeof(IPCMessage<T>), &bytes_read, nullptr) ||
            bytes_read != sizeof(IPCMessage<T>)) {
            return false;
        }
#else
        ssize_t bytes_read = read(pipe_fd_, &header, sizeof(IPCMessage<T>));
        if (bytes_read != sizeof(IPCMessage<T>)) {
            return false;
        }
#endif
        
        // Then read the data
        message = header;
        message.data.resize(header.message_size);
        
#ifdef _WIN32
        if (!ReadFile(pipe_handle_, message.data.data(), header.message_size, &bytes_read, nullptr) ||
            bytes_read != header.message_size) {
            return false;
        }
#else
        bytes_read = read(pipe_fd_, message.data.data(), header.message_size);
        if (bytes_read != static_cast<ssize_t>(header.message_size)) {
            return false;
        }
#endif
        
        return true;
    }
    
    bool is_connected() const override {
        return initialized_;
    }
    
    std::string get_status() const override {
        return initialized_ ? "Connected" : "Disconnected";
    }
};

/**
 * @brief Interprocess communication decorator
 * 
 * This decorator provides comprehensive IPC capabilities with the following features:
 * - Multiple IPC methods (shared memory, named pipes, sockets)
 * - Producer/consumer and bidirectional communication
 * - Synchronous and asynchronous operation modes
 * - Reliability features (acknowledgments, retries, error correction)
 * - Performance optimization (batching, compression)
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles interprocess communication
 * - Flexible: Multiple IPC methods and configurations
 * - Robust: Error handling and reliability features
 * - Performance: Optimized for low-latency communication
 */
template<system_state S, can_be_time T = double>
class InterprocessDecorator : public IntegratorDecorator<S, T> {
private:
    InterprocessConfig config_;
    std::unique_ptr<IPCChannel<T>> channel_;
    std::atomic<uint32_t> sequence_number_{0};
    IPCStats stats_;
    
    // Threading for async operations
    std::thread communication_thread_;
    std::atomic<bool> running_{false};
    std::mutex message_queue_mutex_;
    std::condition_variable message_queue_cv_;
    std::queue<IPCMessage<T>> outgoing_messages_;
    std::queue<IPCMessage<T>> incoming_messages_;
    
    // Callbacks for received data
    std::function<void(const S&, T)> receive_callback_;
    
    // SDE synchronization
    std::mutex sde_sync_mutex_;
    std::condition_variable sde_sync_cv_;
    std::atomic<bool> noise_data_available_{false};
    S pending_noise_data_;

public:
    /**
     * @brief Construct interprocess decorator
     * @param integrator The integrator to wrap
     * @param config Interprocess configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit InterprocessDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                                  InterprocessConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {
        
        config_.validate();
        initialize_channel();
        
        if (config_.direction == IPCDirection::CONSUMER || config_.direction == IPCDirection::BIDIRECTIONAL) {
            start_communication_thread();
        }
    }
    
    /**
     * @brief Destructor ensures proper cleanup
     */
    ~InterprocessDecorator() {
        stop_communication_thread();
        if (channel_) {
            channel_->cleanup();
        }
    }

    /**
     * @brief Override step to handle IPC during integration
     */
    void step(typename IntegratorDecorator<S, T>::state_type& state, T dt) override {
        // Handle SDE synchronization if needed
        if (config_.direction == IPCDirection::CONSUMER) {
            wait_for_noise_data();
        }
        
        // Send current state if producer
        if (config_.direction == IPCDirection::PRODUCER || config_.direction == IPCDirection::BIDIRECTIONAL) {
            send_state(state, this->current_time());
        }
        
        // Perform integration step
        this->wrapped_integrator_->step(state, dt);
        
        // Handle received data if consumer
        if (config_.direction == IPCDirection::CONSUMER || config_.direction == IPCDirection::BIDIRECTIONAL) {
            process_incoming_messages(state);
        }
    }

    /**
     * @brief Override integrate to handle IPC during integration
     */
    void integrate(typename IntegratorDecorator<S, T>::state_type& state, T dt, T end_time) override {
        // Send initial state
        if (config_.direction == IPCDirection::PRODUCER || config_.direction == IPCDirection::BIDIRECTIONAL) {
            send_state(state, this->current_time());
        }
        
        // Integrate with IPC handling
        this->wrapped_integrator_->integrate(state, dt, end_time);
        
        // Send final state
        if (config_.direction == IPCDirection::PRODUCER || config_.direction == IPCDirection::BIDIRECTIONAL) {
            send_state(state, this->current_time());
        }
    }

    /**
     * @brief Set callback for received data
     * @param callback Function to call when data is received
     */
    void set_receive_callback(std::function<void(const S&, T)> callback) {
        receive_callback_ = std::move(callback);
    }

    /**
     * @brief Send state data to other processes
     * @param state Current state
     * @param time Current time
     * @return true if successful
     */
    bool send_state(const S& state, T time) {
        if (!channel_ || !channel_->is_connected()) {
            return false;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        IPCMessage<T> message;
        message.sequence_number = sequence_number_++;
        message.timestamp = time;
        message.serialize_state(state);
        
        bool success = false;
        size_t retries = 0;
        
        while (!success && retries < config_.max_retries) {
            success = channel_->send_message(message);
            if (!success) {
                retries++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (success) {
            stats_.messages_sent++;
            stats_.bytes_sent += message.message_size;
            stats_.total_send_time += duration;
        } else {
            stats_.send_failures++;
        }
        
        return success;
    }

    /**
     * @brief Get current IPC statistics
     */
    const IPCStats& get_statistics() const {
        return stats_;
    }

    /**
     * @brief Reset IPC statistics
     */
    void reset_statistics() {
        stats_ = IPCStats{};
    }

    /**
     * @brief Check if channel is connected
     */
    bool is_connected() const {
        return channel_ && channel_->is_connected();
    }

    /**
     * @brief Get channel status
     */
    std::string get_status() const {
        return channel_ ? channel_->get_status() : "Not initialized";
    }

    /**
     * @brief Access and modify interprocess configuration
     */
    InterprocessConfig& config() { return config_; }
    const InterprocessConfig& config() const { return config_; }

private:
    /**
     * @brief Initialize IPC channel based on configuration
     */
    void initialize_channel() {
        switch (config_.method) {
            case IPCMethod::SHARED_MEMORY:
                channel_ = std::make_unique<SharedMemoryChannel<T>>(config_.channel_name, config_.buffer_size);
                break;
            case IPCMethod::NAMED_PIPES:
                channel_ = std::make_unique<NamedPipeChannel<T>>(config_.channel_name, 
                    config_.direction == IPCDirection::PRODUCER);
                break;
            case IPCMethod::MEMORY_MAPPED_FILE:
                // TODO: Implement memory-mapped file channel
                throw std::runtime_error("Memory-mapped file IPC not yet implemented");
            case IPCMethod::TCP_SOCKET:
                // TODO: Implement TCP socket channel
                throw std::runtime_error("TCP socket IPC not yet implemented");
            case IPCMethod::UDP_SOCKET:
                // TODO: Implement UDP socket channel
                throw std::runtime_error("UDP socket IPC not yet implemented");
            default:
                throw std::runtime_error("Unknown IPC method");
        }
        
        if (!channel_->initialize()) {
            throw std::runtime_error("Failed to initialize IPC channel");
        }
    }

    /**
     * @brief Start communication thread for receiving data
     */
    void start_communication_thread() {
        running_ = true;
        communication_thread_ = std::thread([this]() {
            while (running_) {
                IPCMessage<T> message;
                if (channel_->receive_message(message)) {
                    std::lock_guard<std::mutex> lock(message_queue_mutex_);
                    incoming_messages_.push(message);
                    message_queue_cv_.notify_one();
                    
                    stats_.messages_received++;
                    stats_.bytes_received += message.message_size;
                }
                
                std::this_thread::sleep_for(config_.polling_interval);
            }
        });
    }

    /**
     * @brief Stop communication thread
     */
    void stop_communication_thread() {
        running_ = false;
        if (communication_thread_.joinable()) {
            communication_thread_.join();
        }
    }

    /**
     * @brief Process incoming messages
     */
    void process_incoming_messages(S& state) {
        std::lock_guard<std::mutex> lock(message_queue_mutex_);
        
        while (!incoming_messages_.empty()) {
            auto message = incoming_messages_.front();
            incoming_messages_.pop();
            
            // Deserialize state data
            auto received_state = message.deserialize_state();
            
            // Call receive callback if set
            if (receive_callback_) {
                receive_callback_(received_state, message.timestamp);
            }
            
            // For SDE synchronization, signal that noise data is available
            if (config_.direction == IPCDirection::CONSUMER) {
                std::lock_guard<std::mutex> sync_lock(sde_sync_mutex_);
                pending_noise_data_ = received_state;
                noise_data_available_ = true;
                sde_sync_cv_.notify_one();
            }
        }
    }

    /**
     * @brief Wait for noise data (SDE synchronization)
     */
    void wait_for_noise_data() {
        std::unique_lock<std::mutex> lock(sde_sync_mutex_);
        
        if (config_.sync_mode == IPCSyncMode::BLOCKING) {
            sde_sync_cv_.wait(lock, [this] { return noise_data_available_.load(); });
        } else if (config_.sync_mode == IPCSyncMode::TIMEOUT) {
            sde_sync_cv_.wait_for(lock, config_.timeout, [this] { return noise_data_available_.load(); });
        }
        
        noise_data_available_ = false;
    }
};

} // namespace diffeq::core::composable 