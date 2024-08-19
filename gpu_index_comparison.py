import numpy as np
import faiss
import time

# 创建示例数据
d = 128  # 向量维度
nb = 100000  # 数据库向量数量
nq = 1000  # 查询向量数量

np.random.seed(1234)
database_vectors = np.random.random((nb, d)).astype('float32')
query_vectors = np.random.random((nq, d)).astype('float32')

# 使用 GPU 上的 IndexFlatL2
print("Using IndexFlatL2 on GPU")
gpu_res = faiss.StandardGpuResources()  # 初始化 GPU 资源
index_flat_cpu = faiss.IndexFlatL2(d)  # 在 CPU 上创建索引
index_flat_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_flat_cpu)  # 将索引传输到 GPU

start_time = time.time()
index_flat_gpu.add(database_vectors)
flat_add_time = time.time() - start_time

start_time = time.time()
distances_flat, indices_flat = index_flat_gpu.search(query_vectors, 5)
flat_search_time = time.time() - start_time

print(f"IndexFlatL2 (GPU) add time: {flat_add_time:.4f} seconds")
print(f"IndexFlatL2 (GPU) search time: {flat_search_time:.4f} seconds")

# 使用 GPU 上的 HNSW
print("\nUsing HNSW on GPU")
m = 200  # 每个节点的最大连接数，增大以提高精度
index_hnsw_cpu = faiss.IndexHNSWFlat(d, m)

# 将 HNSW 索引传输到 GPU
index_hnsw_gpu = faiss.index_cpu_to_gpu(gpu_res, 0, index_hnsw_cpu)

start_time = time.time()
index_hnsw_gpu.add(database_vectors)
hnsw_add_time = time.time() - start_time

# 设置 efSearch 参数，在搜索时访问更多的节点
index_hnsw_gpu.hnsw.efSearch = 300

start_time = time.time()
distances_hnsw, indices_hnsw = index_hnsw_gpu.search(query_vectors, 5)
hnsw_search_time = time.time() - start_time

print(f"HNSW (GPU) add time: {hnsw_add_time:.4f} seconds")
print(f"HNSW (GPU) search time: {hnsw_search_time:.4f} seconds")

# 验证搜索结果是否一致（简单验证输出前5个查询的结果）
print("\nComparing search results:")
if np.allclose(distances_flat, distances_hnsw, atol=1e-6) and np.array_equal(indices_flat, indices_hnsw):
    print("The search results from IndexFlatL2 (GPU) and HNSW (GPU) are the same.")
else:
    print("The search results from IndexFlatL2 (GPU) and HNSW (GPU) are different.")
