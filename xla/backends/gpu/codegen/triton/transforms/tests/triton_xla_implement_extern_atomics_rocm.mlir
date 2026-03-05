// RUN: xla-opt %s -triton-xla-implement-extern-atomics-rocm | FileCheck %s

// Test ROCm implementation of extern_elementwise atomic functions
// This pass operates on LLVM dialect and inlines the implementations

// Test unmasked operations
module {
  // CHECK-LABEL: llvm.func @test_get_thread_id
  llvm.func @test_get_thread_id() -> i32 {
    // CHECK-NOT: llvm.call @xla_get_thread_id
    // CHECK: [[TID:%.*]] = llvm.call_intrinsic "llvm.amdgcn.workitem.id.x"() : () -> i32
    // CHECK: llvm.return [[TID]]
    %tid = llvm.call @xla_get_thread_id() : () -> i32
    llvm.return %tid : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_write_unmasked
  llvm.func @test_atomic_write_unmasked(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomic_write_release_system
    // CHECK: [[RESULT:%.*]] = llvm.atomicrmw xchg %arg0, %arg1 release : !llvm.ptr<1>, i32
    // CHECK: llvm.return [[RESULT]]
    %result = llvm.call @xla_atomic_write_release_system(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_unmasked
  llvm.func @test_atomic_spin_wait_unmasked(%ptr: !llvm.ptr<1>, %expected: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomic_spin_wait_acquire_system_lt
    // CHECK: llvm.br ^[[LOOP:.*]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "ult" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[LOOP]], ^[[EXIT:.*]]([[LOADED]]
    // CHECK: ^[[EXIT]]([[RESULT:%.*]]: i32):
    // CHECK:   llvm.return [[RESULT]]
    %result = llvm.call @xla_atomic_spin_wait_acquire_system_lt(%ptr, %expected) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_relaxed_ordering
  llvm.func @test_relaxed_ordering(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: llvm.atomicrmw xchg %arg0, %arg1 monotonic : !llvm.ptr<1>, i32
    %result = llvm.call @xla_atomic_write_relaxed_system(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_agent_scope
  llvm.func @test_agent_scope(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: llvm.atomicrmw xchg %arg0, %arg1 syncscope("agent") release : !llvm.ptr<1>, i32
    %result = llvm.call @xla_atomic_write_release_gpu(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_workgroup_scope
  llvm.func @test_workgroup_scope(%ptr: !llvm.ptr<1>, %value: i32) -> i32 {
    // CHECK: llvm.atomicrmw xchg %arg0, %arg1 syncscope("workgroup") release : !llvm.ptr<1>, i32
    %result = llvm.call @xla_atomic_write_release_cta(%ptr, %value) : (!llvm.ptr<1>, i32) -> i32
    llvm.return %result : i32
  }
  
  // Extern function declarations (will be removed by the pass)
  llvm.func @xla_get_thread_id() -> i32
  llvm.func @xla_atomic_write_release_system(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomic_write_relaxed_system(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomic_write_release_gpu(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomic_write_release_cta(!llvm.ptr<1>, i32) -> i32
  llvm.func @xla_atomic_spin_wait_acquire_system_lt(!llvm.ptr<1>, i32) -> i32
}

// Test masked operations in separate module to avoid function redefinition
module {
  // CHECK-LABEL: llvm.func @test_atomic_write_masked
  llvm.func @test_atomic_write_masked(%ptr: !llvm.ptr<1>, %value: i32, %mask: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomic_write_release_system
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[MASK_NONZERO:%.*]] = llvm.icmp "ne" %arg2, [[ZERO]]
    // CHECK: llvm.cond_br [[MASK_NONZERO]], ^[[ATOMIC:.*]], ^[[SKIP:.*]]
    // CHECK: ^[[ATOMIC]]:
    // CHECK:   [[ATOMIC_RESULT:%.*]] = llvm.atomicrmw xchg %arg0, %arg1 release : !llvm.ptr<1>, i32
    // CHECK:   llvm.br ^[[EXIT:.*]]([[ATOMIC_RESULT]]
    // CHECK: ^[[SKIP]]:
    // CHECK:   llvm.br ^[[EXIT]]([[ZERO]]
    // CHECK: ^[[EXIT]]([[PHI:%.*]]: i32):
    // CHECK:   llvm.return [[PHI]]
    %result = llvm.call @xla_atomic_write_release_system(%ptr, %value, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // CHECK-LABEL: llvm.func @test_atomic_spin_wait_masked
  llvm.func @test_atomic_spin_wait_masked(%ptr: !llvm.ptr<1>, %expected: i32, %mask: i32) -> i32 {
    // CHECK-NOT: llvm.call @xla_atomic_spin_wait_acquire_system_lt
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: [[MASK_NONZERO:%.*]] = llvm.icmp "ne" %arg2, [[ZERO]]
    // CHECK: llvm.cond_br [[MASK_NONZERO]], ^[[LOOP:.*]], ^[[EXIT:.*]]([[ZERO]]
    // CHECK: ^[[LOOP]]:
    // CHECK:   [[LOADED:%.*]] = llvm.load %arg0 atomic acquire {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK:   [[COND:%.*]] = llvm.icmp "ult" [[LOADED]], %arg1
    // CHECK:   llvm.cond_br [[COND]], ^[[LOOP]], ^[[EXIT]]([[LOADED]]
    // CHECK: ^[[EXIT]]([[RESULT:%.*]]: i32):
    // CHECK:   llvm.return [[RESULT]]
    %result = llvm.call @xla_atomic_spin_wait_acquire_system_lt(%ptr, %expected, %mask) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.return %result : i32
  }
  
  // Extern function declarations for masked versions
  llvm.func @xla_atomic_write_release_system(!llvm.ptr<1>, i32, i32) -> i32
  llvm.func @xla_atomic_spin_wait_acquire_system_lt(!llvm.ptr<1>, i32, i32) -> i32
}
