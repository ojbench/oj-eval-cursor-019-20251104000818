#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  
  // Create copies of keys and values in SRAM for fast concatenation
  std::vector<Matrix*> keys_sram(keys.size());
  std::vector<Matrix*> values_sram(values.size());
  
  for (size_t i = 0; i < keys.size(); ++i) {
    keys_sram[i] = matrix_memory_allocator.Allocate("key_sram");
    gpu_sim.Copy(keys[i], keys_sram[i], kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(keys_sram[i]);
    
    values_sram[i] = matrix_memory_allocator.Allocate("value_sram");
    gpu_sim.Copy(values[i], values_sram[i], kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(values_sram[i]);
  }
  
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    gpu_sim.MoveMatrixToSharedMem(current_query);
    
    // Build K_all in SRAM (25x faster than HBM)
    Matrix* K_all = matrix_memory_allocator.Allocate("K_all");
    gpu_sim.Copy(keys_sram[0], K_all, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* K_all_new = matrix_memory_allocator.Allocate("K_all_new");
      gpu_sim.Concat(K_all, keys_sram[j], K_all_new, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K_all);
      K_all = K_all_new;
    }
    
    // Build V_all in SRAM (25x faster than HBM)
    Matrix* V_all = matrix_memory_allocator.Allocate("V_all");
    gpu_sim.Copy(values_sram[0], V_all, kInSharedMemory);
    for (size_t j = 1; j <= i; ++j) {
      Matrix* V_all_new = matrix_memory_allocator.Allocate("V_all_new");
      gpu_sim.Concat(V_all, values_sram[j], V_all_new, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V_all);
      V_all = V_all_new;
    }
    
    // Transpose K_all
    gpu_sim.Transpose(K_all, kInSharedMemory);
    
    // Compute Q x K_all^T
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K_all, QK);
    gpu_sim.ReleaseMatrix(K_all);
    
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
    
    // Compute softmax_matrix x V_all
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_matrix, V_all, result);
    gpu_sim.ReleaseMatrix(softmax_matrix);
    gpu_sim.ReleaseMatrix(V_all);
    
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
