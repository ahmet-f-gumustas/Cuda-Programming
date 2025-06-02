#include "../include/common.h"

// Graf yapısı (CSR format)
struct Graph {
    int num_vertices;
    int num_edges;
    std::vector<int> row_offsets;  // Her vertex'in adjacency list'inin başlangıcı
    std::vector<int> column_indices; // Komşu vertex'ler
    
    Graph(int v, int e) : num_vertices(v), num_edges(e) {
        row_offsets.resize(v + 1);
        column_indices.resize(e);
    }
};

// Random graf oluştur
Graph generate_random_graph(int num_vertices, int avg_degree) {
    int num_edges = num_vertices * avg_degree;
    Graph g(num_vertices, num_edges);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_vertices - 1);
    
    // Her vertex için edge sayısını belirle
    std::vector<std::vector<int>> adj_lists(num_vertices);
    
    for (int i = 0; i < num_vertices; ++i) {
        int degree = avg_degree + (gen() % 3) - 1; // avg_degree ± 1
        degree = std::max(1, std::min(degree, num_vertices - 1));
        
        std::set<int> neighbors;
        while (neighbors.size() < static_cast<size_t>(degree)) {
            int neighbor = dis(gen);
            if (neighbor != i) {
                neighbors.insert(neighbor);
            }
        }
        
        adj_lists[i] = std::vector<int>(neighbors.begin(), neighbors.end());
    }
    
    // CSR format'a dönüştür
    int edge_count = 0;
    g.row_offsets[0] = 0;
    
    for (int i = 0; i < num_vertices; ++i) {
        for (int neighbor : adj_lists[i]) {
            g.column_indices[edge_count++] = neighbor;
        }
        g.row_offsets[i + 1] = edge_count;
    }
    
    g.num_edges = edge_count;
    g.column_indices.resize(edge_count);
    
    return g;
}

// Naive BFS kernel
__global__ void naive_bfs_kernel(const int* row_offsets, const int* column_indices,
                                int* distances, bool* visited, bool* current_frontier,
                                bool* next_frontier, int num_vertices, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices && current_frontier[tid]) {
        current_frontier[tid] = false;
        
        int start = row_offsets[tid];
        int end = row_offsets[tid + 1];
        
        for (int i = start; i < end; ++i) {
            int neighbor = column_indices[i];
            if (!visited[neighbor]) {
                if (atomicCAS(&visited[neighbor], false, true) == false) {
                    distances[neighbor] = level + 1;
                    next_frontier[neighbor] = true;
                }
            }
        }
    }
}

// Shared memory optimization ile BFS
__global__ void shared_bfs_kernel(const int* row_offsets, const int* column_indices,
                                 int* distances, bool* visited, bool* current_frontier,
                                 bool* next_frontier, int num_vertices, int level) {
    extern __shared__ bool shared_frontier[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Shared memory'ye frontier bilgisini yükle
    if (tid < num_vertices) {
        shared_frontier[local_tid] = current_frontier[tid];
    } else {
        shared_frontier[local_tid] = false;
    }
    __syncthreads();
    
    if (tid < num_vertices && shared_frontier[local_tid]) {
        current_frontier[tid] = false;
        
        int start = row_offsets[tid];
        int end = row_offsets[tid + 1];
        
        for (int i = start; i < end; ++i) {
            int neighbor = column_indices[i];
            if (!visited[neighbor]) {
                if (atomicCAS(&visited[neighbor], false, true) == false) {
                    distances[neighbor] = level + 1;
                    next_frontier[neighbor] = true;
                }
            }
        }
    }
}

// Warp-level cooperation ile BFS
__global__ void warp_bfs_kernel(const int* row_offsets, const int* column_indices,
                               int* distances, bool* visited, bool* current_frontier,
                               bool* next_frontier, int num_vertices, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Her warp bir vertex işler
    int vertex = warp_id;
    
    if (vertex < num_vertices && current_frontier[vertex]) {
        if (lane_id == 0) {
            current_frontier[vertex] = false;
        }
        
        int start = row_offsets[vertex];
        int end = row_offsets[vertex + 1];
        int degree = end - start;
        
        // Warp içindeki thread'ler komşuları parallel işler
        for (int i = lane_id; i < degree; i += WARP_SIZE) {
            int neighbor = column_indices[start + i];
            if (!visited[neighbor]) {
                if (atomicCAS(&visited[neighbor], false, true) == false) {
                    distances[neighbor] = level + 1;
                    next_frontier[neighbor] = true;
                }
            }
        }
    }
}

// Work-efficient BFS with dynamic frontier
__global__ void workefficient_bfs_kernel(const int* row_offsets, const int* column_indices,
                                        int* distances, bool* visited, int* current_frontier,
                                        int* next_frontier, int* frontier_size,
                                        int* next_frontier_size, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < *frontier_size) {
        int vertex = current_frontier[tid];
        
        int start = row_offsets[vertex];
        int end = row_offsets[vertex + 1];
        
        for (int i = start; i < end; ++i) {
            int neighbor = column_indices[i];
            if (!visited[neighbor]) {
                if (atomicCAS(&visited[neighbor], false, true) == false) {
                    distances[neighbor] = level + 1;
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = neighbor;
                }
            }
        }
    }
}

// Direction-optimizing BFS (top-down + bottom-up)
__global__ void bottom_up_bfs_kernel(const int* row_offsets, const int* column_indices,
                                    int* distances, bool* visited, bool* current_frontier,
                                    bool* next_frontier, int num_vertices, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices && !visited[tid]) {
        int start = row_offsets[tid];
        int end = row_offsets[tid + 1];
        
        // Bu vertex'in komşularında frontier var mı kontrol et
        for (int i = start; i < end; ++i) {
            int neighbor = column_indices[i];
            if (current_frontier[neighbor]) {
                visited[tid] = true;
                distances[tid] = level + 1;
                next_frontier[tid] = true;
                break;
            }
        }
    }
}

// Ana BFS fonksiyonu
void bfs_custom(const Graph& g, int source, std::vector<int>& distances, const std::string& method) {
    int num_vertices = g.num_vertices;
    distances.assign(num_vertices, -1);
    distances[source] = 0;
    
    // Device memory ayır
    ManagedMemory<int> d_row_offsets(num_vertices + 1);
    ManagedMemory<int> d_column_indices(g.num_edges);
    ManagedMemory<int> d_distances(num_vertices);
    ManagedMemory<bool> d_visited(num_vertices);
    ManagedMemory<bool> d_current_frontier(num_vertices);
    ManagedMemory<bool> d_next_frontier(num_vertices);
    
    // Host'tan device'a kopyala
    d_row_offsets.copy_from_host(g.row_offsets.data());
    d_column_indices.copy_from_host(g.column_indices.data());
    d_distances.copy_from_host(distances.data());
    
    // Initialize arrays
    CUDA_CHECK(cudaMemset(d_visited.get(), false, num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_current_frontier.get(), false, num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_next_frontier.get(), false, num_vertices * sizeof(bool)));
    
    // Source vertex'i işaretle
    bool source_true = true;
    CUDA_CHECK(cudaMemcpy(d_visited.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_current_frontier.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
    
    CudaTimer timer;
    timer.start();
    
    int level = 0;
    bool frontier_not_empty = true;
    
    while (frontier_not_empty) {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (num_vertices + threads_per_block - 1) / threads_per_block;
        
        if (method == "naive") {
            naive_bfs_kernel<<<blocks, threads_per_block>>>(
                d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
                d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
                num_vertices, level);
        }
        else if (method == "shared") {
            int shared_mem_size = threads_per_block * sizeof(bool);
            shared_bfs_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
                d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
                num_vertices, level);
        }
        else if (method == "warp") {
            int total_warps = (num_vertices + WARP_SIZE - 1) / WARP_SIZE;
            int warps_per_block = threads_per_block / WARP_SIZE;
            int blocks_for_warps = (total_warps + warps_per_block - 1) / warps_per_block;
            
            warp_bfs_kernel<<<blocks_for_warps, threads_per_block>>>(
                d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
                d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
                num_vertices, level);
        }
        
        CUDA_CHECK_KERNEL();
        
        // Frontier'ları swap et
        std::swap(d_current_frontier, d_next_frontier);
        
        // Next frontier'ı temizle
        CUDA_CHECK(cudaMemset(d_next_frontier.get(), false, num_vertices * sizeof(bool)));
        
        // Frontier boş mu kontrol et
        std::vector<bool> h_current_frontier(num_vertices);
        d_current_frontier.copy_to_host(h_current_frontier.data());
        
        frontier_not_empty = false;
        for (bool val : h_current_frontier) {
            if (val) {
                frontier_not_empty = true;
                break;
            }
        }
        
        level++;
        
        if (level > num_vertices) { // Sonsuz döngü koruması
            break;
        }
    }
    
    timer.stop();
    
    d_distances.copy_to_host(distances.data());
    
    std::cout << "Custom BFS (" << method << ") - Time: " 
              << timer.elapsed_ms() << " ms, Levels: " << level << std::endl;
}

// Work-efficient BFS implementation
void bfs_work_efficient(const Graph& g, int source, std::vector<int>& distances) {
    int num_vertices = g.num_vertices;
    distances.assign(num_vertices, -1);
    distances[source] = 0;
    
    // Device memory ayır
    ManagedMemory<int> d_row_offsets(num_vertices + 1);
    ManagedMemory<int> d_column_indices(g.num_edges);
    ManagedMemory<int> d_distances(num_vertices);
    ManagedMemory<bool> d_visited(num_vertices);
    ManagedMemory<int> d_current_frontier(num_vertices);
    ManagedMemory<int> d_next_frontier(num_vertices);
    ManagedMemory<int> d_frontier_size(1);
    ManagedMemory<int> d_next_frontier_size(1);
    
    // Host'tan device'a kopyala
    d_row_offsets.copy_from_host(g.row_offsets.data());
    d_column_indices.copy_from_host(g.column_indices.data());
    d_distances.copy_from_host(distances.data());
    
    // Initialize arrays
    CUDA_CHECK(cudaMemset(d_visited.get(), false, num_vertices * sizeof(bool)));
    
    // Source vertex'i ayarla
    bool source_true = true;
    int initial_frontier_size = 1;
    CUDA_CHECK(cudaMemcpy(d_visited.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_current_frontier.get(), &source, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_size.get(), &initial_frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    
    CudaTimer timer;
    timer.start();
    
    int level = 0;
    int h_frontier_size = 1;
    
    while (h_frontier_size > 0) {
        // Next frontier size'ı sıfırla
        CUDA_CHECK(cudaMemset(d_next_frontier_size.get(), 0, sizeof(int)));
        
        int threads_per_block = BLOCK_SIZE;
        int blocks = (h_frontier_size + threads_per_block - 1) / threads_per_block;
        
        workefficient_bfs_kernel<<<blocks, threads_per_block>>>(
            d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
            d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
            d_frontier_size.get(), d_next_frontier_size.get(), level);
        
        CUDA_CHECK_KERNEL();
        
        // Frontier'ları swap et
        std::swap(d_current_frontier, d_next_frontier);
        std::swap(d_frontier_size, d_next_frontier_size);
        
        // Yeni frontier size'ı al
        d_frontier_size.copy_to_host(&h_frontier_size);
        
        level++;
        
        if (level > num_vertices) { // Sonsuz döngü koruması
            break;
        }
    }
    
    timer.stop();
    
    d_distances.copy_to_host(distances.data());
    
    std::cout << "Work-efficient BFS - Time: " 
              << timer.elapsed_ms() << " ms, Levels: " << level << std::endl;
}

// Direction-optimizing BFS
void bfs_direction_optimizing(const Graph& g, int source, std::vector<int>& distances) {
    int num_vertices = g.num_vertices;
    distances.assign(num_vertices, -1);
    distances[source] = 0;
    
    // Device memory ayır
    ManagedMemory<int> d_row_offsets(num_vertices + 1);
    ManagedMemory<int> d_column_indices(g.num_edges);
    ManagedMemory<int> d_distances(num_vertices);
    ManagedMemory<bool> d_visited(num_vertices);
    ManagedMemory<bool> d_current_frontier(num_vertices);
    ManagedMemory<bool> d_next_frontier(num_vertices);
    
    // Host'tan device'a kopyala
    d_row_offsets.copy_from_host(g.row_offsets.data());
    d_column_indices.copy_from_host(g.column_indices.data());
    d_distances.copy_from_host(distances.data());
    
    // Initialize arrays
    CUDA_CHECK(cudaMemset(d_visited.get(), false, num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_current_frontier.get(), false, num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_next_frontier.get(), false, num_vertices * sizeof(bool)));
    
    // Source vertex'i işaretle
    bool source_true = true;
    CUDA_CHECK(cudaMemcpy(d_visited.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_current_frontier.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
    
    CudaTimer timer;
    timer.start();
    
    int level = 0;
    bool frontier_not_empty = true;
    
    while (frontier_not_empty) {
        // Frontier size'ı tahmin et
        std::vector<bool> h_current_frontier(num_vertices);
        d_current_frontier.copy_to_host(h_current_frontier.data());
        
        int frontier_size = 0;
        for (bool val : h_current_frontier) {
            if (val) frontier_size++;
        }
        
        int threads_per_block = BLOCK_SIZE;
        int blocks = (num_vertices + threads_per_block - 1) / threads_per_block;
        
        // Threshold'a göre direction seç
        // Küçük frontier için top-down, büyük frontier için bottom-up
        if (frontier_size < num_vertices / 20) {
            // Top-down BFS
            naive_bfs_kernel<<<blocks, threads_per_block>>>(
                d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
                d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
                num_vertices, level);
        } else {
            // Bottom-up BFS
            bottom_up_bfs_kernel<<<blocks, threads_per_block>>>(
                d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
                d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
                num_vertices, level);
        }
        
        CUDA_CHECK_KERNEL();
        
        // Frontier'ları swap et
        std::swap(d_current_frontier, d_next_frontier);
        
        // Next frontier'ı temizle
        CUDA_CHECK(cudaMemset(d_next_frontier.get(), false, num_vertices * sizeof(bool)));
        
        // Frontier boş mu kontrol et
        d_current_frontier.copy_to_host(h_current_frontier.data());
        
        frontier_not_empty = false;
        for (bool val : h_current_frontier) {
            if (val) {
                frontier_not_empty = true;
                break;
            }
        }
        
        level++;
        
        if (level > num_vertices) { // Sonsuz döngü koruması
            break;
        }
    }
    
    timer.stop();
    
    d_distances.copy_to_host(distances.data());
    
    std::cout << "Direction-optimizing BFS - Time: " 
              << timer.elapsed_ms() << " ms, Levels: " << level << std::endl;
}

// CPU referans BFS
void bfs_cpu(const Graph& g, int source, std::vector<int>& distances) {
    distances.assign(g.num_vertices, -1);
    distances[source] = 0;
    
    std::queue<int> frontier;
    frontier.push(source);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    while (!frontier.empty()) {
        int vertex = frontier.front();
        frontier.pop();
        
        int start_edge = g.row_offsets[vertex];
        int end_edge = g.row_offsets[vertex + 1];
        
        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor = g.column_indices[i];
            if (distances[neighbor] == -1) {
                distances[neighbor] = distances[vertex] + 1;
                frontier.push(neighbor);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    std::cout << "CPU BFS - Time: " << cpu_time << " ms" << std::endl;
}

// Multi-source BFS
__global__ void multi_source_bfs_kernel(const int* row_offsets, const int* column_indices,
                                       int* distances, bool* visited, bool* current_frontier,
                                       bool* next_frontier, int num_vertices, int level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vertices && current_frontier[tid]) {
        current_frontier[tid] = false;
        
        int start = row_offsets[tid];
        int end = row_offsets[tid + 1];
        
        for (int i = start; i < end; ++i) {
            int neighbor = column_indices[i];
            if (!visited[neighbor]) {
                if (atomicCAS(&visited[neighbor], false, true) == false) {
                    distances[neighbor] = level + 1;
                    next_frontier[neighbor] = true;
                }
            }
        }
    }
}

// Multi-source BFS implementation
void bfs_multi_source(const Graph& g, const std::vector<int>& sources, std::vector<int>& distances) {
    int num_vertices = g.num_vertices;
    distances.assign(num_vertices, -1);
    
    // Device memory ayır
    ManagedMemory<int> d_row_offsets(num_vertices + 1);
    ManagedMemory<int> d_column_indices(g.num_edges);
    ManagedMemory<int> d_distances(num_vertices);
    ManagedMemory<bool> d_visited(num_vertices);
    ManagedMemory<bool> d_current_frontier(num_vertices);
    ManagedMemory<bool> d_next_frontier(num_vertices);
    
    // Host'tan device'a kopyala
    d_row_offsets.copy_from_host(g.row_offsets.data());
    d_column_indices.copy_from_host(g.column_indices.data());
    d_distances.copy_from_host(distances.data());
    
    // Initialize arrays
    CUDA_CHECK(cudaMemset(d_visited.get(), false, num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_current_frontier.get(), false, num_vertices * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_next_frontier.get(), false, num_vertices * sizeof(bool)));
    
    // Source vertex'leri işaretle
    for (int source : sources) {
        distances[source] = 0;
        bool source_true = true;
        CUDA_CHECK(cudaMemcpy(d_visited.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_current_frontier.get() + source, &source_true, sizeof(bool), cudaMemcpyHostToDevice));
    }
    
    // Distances'ı tekrar kopyala
    d_distances.copy_from_host(distances.data());
    
    CudaTimer timer;
    timer.start();
    
    int level = 0;
    bool frontier_not_empty = true;
    
    while (frontier_not_empty) {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (num_vertices + threads_per_block - 1) / threads_per_block;
        
        multi_source_bfs_kernel<<<blocks, threads_per_block>>>(
            d_row_offsets.get(), d_column_indices.get(), d_distances.get(),
            d_visited.get(), d_current_frontier.get(), d_next_frontier.get(),
            num_vertices, level);
        
        CUDA_CHECK_KERNEL();
        
        // Frontier'ları swap et
        std::swap(d_current_frontier, d_next_frontier);
        
        // Next frontier'ı temizle
        CUDA_CHECK(cudaMemset(d_next_frontier.get(), false, num_vertices * sizeof(bool)));
        
        // Frontier boş mu kontrol et
        std::vector<bool> h_current_frontier(num_vertices);
        d_current_frontier.copy_to_host(h_current_frontier.data());
        
        frontier_not_empty = false;
        for (bool val : h_current_frontier) {
            if (val) {
                frontier_not_empty = true;
                break;
            }
        }
        
        level++;
        
        if (level > num_vertices) { // Sonsuz döngü koruması
            break;
        }
    }
    
    timer.stop();
    
    d_distances.copy_to_host(distances.data());
    
    std::cout << "Multi-source BFS (" << sources.size() << " sources) - Time: " 
              << timer.elapsed_ms() << " ms, Levels: " << level << std::endl;
}

// Test fonksiyonu
void test_bfs() {
    std::cout << "\n=== BFS ALGORITHM TEST ===" << std::endl;
    
    // Test grafiği oluştur
    const int num_vertices = 10000;
    const int avg_degree = 10;
    
    std::cout << "Generating random graph with " << num_vertices 
              << " vertices and average degree " << avg_degree << "..." << std::endl;
    
    Graph g = generate_random_graph(num_vertices, avg_degree);
    std::cout << "Graph generated: " << g.num_vertices << " vertices, " 
              << g.num_edges << " edges" << std::endl;
    
    int source = 0;
    
    // CPU referans
    std::vector<int> cpu_distances;
    bfs_cpu(g, source, cpu_distances);
    
    // Custom implementasyonlar
    std::vector<int> distances_naive, distances_shared, distances_warp;
    std::vector<int> distances_work_efficient, distances_direction_opt;
    
    bfs_custom(g, source, distances_naive, "naive");
    bfs_custom(g, source, distances_shared, "shared");
    bfs_custom(g, source, distances_warp, "warp");
    bfs_work_efficient(g, source, distances_work_efficient);
    bfs_direction_optimizing(g, source, distances_direction_opt);
    
    // Multi-source BFS test
    std::vector<int> multi_sources = {0, num_vertices/4, num_vertices/2, 3*num_vertices/4};
    std::vector<int> distances_multi_source;
    bfs_multi_source(g, multi_sources, distances_multi_source);
    
    // Doğruluk kontrolü
    bool naive_match = (distances_naive == cpu_distances);
    bool shared_match = (distances_shared == cpu_distances);
    bool warp_match = (distances_warp == cpu_distances);
    bool work_efficient_match = (distances_work_efficient == cpu_distances);
    bool direction_opt_match = (distances_direction_opt == cpu_distances);
    
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << "Naive result matches CPU: " << (naive_match ? "✓" : "✗") << std::endl;
    std::cout << "Shared result matches CPU: " << (shared_match ? "✓" : "✗") << std::endl;
    std::cout << "Warp result matches CPU: " << (warp_match ? "✓" : "✗") << std::endl;
    std::cout << "Work-efficient result matches CPU: " << (work_efficient_match ? "✓" : "✗") << std::endl;
    std::cout << "Direction-optimizing result matches CPU: " << (direction_opt_match ? "✓" : "✗") << std::endl;
    
    // İstatistikler
    int reachable_vertices = 0;
    int max_distance = 0;
    for (int dist : cpu_distances) {
        if (dist != -1) {
            reachable_vertices++;
            max_distance = std::max(max_distance, dist);
        }
    }
    
    std::cout << "\nGraph statistics:" << std::endl;
    std::cout << "Reachable vertices from source " << source << ": " 
              << reachable_vertices << "/" << num_vertices << std::endl;
    std::cout << "Maximum distance: " << max_distance << std::endl;
    
    // İlk 20 vertex'in mesafelerini yazdır
    std::cout << "\nFirst 20 distances:" << std::endl;
    std::cout << "Vertex\tCPU\tGPU" << std::endl;
    for (int i = 0; i < std::min(20, num_vertices); ++i) {
        std::cout << i << "\t" << cpu_distances[i] << "\t" << distances_shared[i] << std::endl;
    }
}