#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  Matrix* K_all = nullptr;
  Matrix* V_all = nullptr;
  
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    
    // Build K_all incrementally
    if (i == 0) {
      K_all = matrix_memory_allocator.Allocate("K_all");
      gpu_sim.Copy(keys[0], K_all, kInGpuHbm);
    } else {
      Matrix* K_all_new = matrix_memory_allocator.Allocate("K_all_new");
      gpu_sim.Concat(K_all, keys[i], K_all_new, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(K_all);
      K_all = K_all_new;
    }
    
    // Build V_all incrementally
    if (i == 0) {
      V_all = matrix_memory_allocator.Allocate("V_all");
      gpu_sim.Copy(values[0], V_all, kInGpuHbm);
    } else {
      Matrix* V_all_new = matrix_memory_allocator.Allocate("V_all_new");
      gpu_sim.Concat(V_all, values[i], V_all_new, 0, kInGpuHbm);
      gpu_sim.ReleaseMatrix(V_all);
      V_all = V_all_new;
    }
    
    // Start IO operations first (move query while we copy)
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Make copies (these are Calculation ops, run in parallel with IO)
    Matrix* K_copy = matrix_memory_allocator.Allocate("K_copy");
    gpu_sim.Copy(K_all, K_copy, kInGpuHbm);
    Matrix* V_copy = matrix_memory_allocator.Allocate("V_copy");
    gpu_sim.Copy(V_all, V_copy, kInGpuHbm);
    
    // Move copies to SRAM
    gpu_sim.MoveMatrixToSharedMem(K_copy);
    gpu_sim.MoveMatrixToSharedMem(V_copy);
    
    // Transpose K_copy
    gpu_sim.Transpose(K_copy, kInSharedMemory);
    
    // Compute Q x K_copy^T
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_copy, QK);
    gpu_sim.ReleaseMatrix(K_copy);
    
    // Apply exp to entire QK matrix
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);
    gpu_sim.ReleaseMatrix(QK);
    
    // Build softmax matrix row by row
    Matrix* softmax_matrix = matrix_memory_allocator.Allocate("softmax_matrix");
    for (size_t row = 0; row <= i; ++row) {
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row");
      gpu_sim.GetRow(QK_exp, row, exp_row, kInSharedMemory);
      
      Matrix* sum_exp = matrix_memory_allocator.Allocate("sum_exp");
      gpu_sim.Sum(exp_row, sum_exp);
      
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);
      
      if (row == 0) {
        gpu_sim.Copy(softmax_row, softmax_matrix, kInSharedMemory);
      } else {
        Matrix* softmax_matrix_new = matrix_memory_allocator.Allocate("softmax_matrix_new");
        gpu_sim.Concat(softmax_matrix, softmax_row, softmax_matrix_new, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_matrix);
        softmax_matrix = softmax_matrix_new;
      }
      gpu_sim.ReleaseMatrix(softmax_row);
    }
    
    gpu_sim.ReleaseMatrix(QK_exp);
    
    // Compute softmax_matrix x V_copy
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_matrix, V_copy, result);
    gpu_sim.ReleaseMatrix(softmax_matrix);
    gpu_sim.ReleaseMatrix(V_copy);
    
    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);
    
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
