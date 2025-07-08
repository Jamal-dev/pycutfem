# pycutfem/jit/__init__.py
from .visitor import IRGenerator
from .codegen import NumbaCodeGen
from .cache import KernelCache
from pycutfem.fem.mixedelement import MixedElement

def compile_backend(integral_expression, mixed_element): # <-- Accept instance
    """
    Orchestrates the full JIT compilation pipeline.
    """
    ir_generator = IRGenerator()
    # Pass the instance to the code generator
    codegen = NumbaCodeGen(mixed_element=mixed_element) 
    cache = KernelCache()

    ir_sequence = ir_generator.generate(integral_expression)
    
    # The cache needs the codegen object to generate source on a cache miss
    kernel, param_order = cache.get_kernel(ir_sequence, codegen)
    
    # Attach py_func for debugging if available
    if hasattr(kernel, "py_func"):
        kernel.python = kernel.py_func
        
    return (kernel, param_order), ir_sequence