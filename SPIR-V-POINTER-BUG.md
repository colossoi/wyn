# SPIR-V Pointer-to-Pointer Bug with Partial Evaluation

## Minimal Reproducer

```wyn
def verts: [3]vec4f32 =
  [@[-1.0, -1.0, 0.0, 1.0],
   @[3.0, -1.0, 0.0, 1.0],
   @[-1.0, 3.0, 0.0, 1.0]]

#[vertex]
def vertex_main(#[builtin(vertex_index)] vertex_id:i32) -> #[builtin(position)] vec4f32 = verts[vertex_id]
```

Compile with `--partial-eval`:
```bash
cargo run --bin wyn -- compile test.wyn --partial-eval -o test.spv
spirv-val test.spv
```

Error:
```
error: In Logical addressing, variables may not allocate a pointer type
  %29 = OpVariable %_ptr_Function__ptr_Function__arr_v4float_int_3 Function
```

## What's Going Wrong

MIR after partial evaluation:

```
locals:
  1: _w_norm_0 (Ptr(Array(Size(3),Vec(Size(4),f32))))   <-- Ptr type!
exprs:
  e16: [e5, e10, e15]        -- array literal
  e17: @materialize(e16)     -- put in memory
  e18: local_1               -- the pointer
  e19: @index(e18, e0)       -- index through pointer
  e20: let local_1 = e17 in e19
```

### Semantics

- `Materialize(e16)` allocates storage for the array, stores the literal, and returns a pointer to the array.
- `local_1` has type `Ptr(Array(...))`, so it holds that pointer value.

### In SPIR-V

1. `Expr::Materialize` lowering creates an `OpVariable` of type `OpTypePointer(Function, %arr_v4float_3)`.

2. Then the generic "declare local" logic sees a MIR local of type `Ptr(Array(...))`, runs it through `ast_type_to_spirv`, and then **again** wraps it in `OpTypePointer(Function, ...)`:

```spirv
%_ptr_Function__arr_v4float_int_3 = OpTypePointer Function %arr_v4float_int_3
%29 = OpVariable %_ptr_Function__ptr_Function__arr_v4float_int_3 Function
```

So `%29` has type `OpTypePointer(Function, %_ptr_Function__arr_v4float_int_3)` - pointer to pointer. That's exactly what the validator complains about.

### Conceptually

- MIR `Ptr(T)` is "an SSA value whose SPIR-V type is `OpTypePointer(Function, T)`".
- SPIR-V `OpVariable` type is already a pointer. You must not wrap pointer types again.

## How to Fix It

Two distinct issues:
1. `ast_type_to_spirv` not handling `Ptr` explicitly (already done at line 298-305)
2. Locals of pointer type going through the "allocate storage" path

### 1. Handle Ptr in ast_type_to_spirv

Already implemented at line 298-305:
```rust
TypeName::Pointer => {
    let pointee_type = self.ast_type_to_spirv(&args[0]);
    self.builder.type_pointer(None, StorageClass::Function, pointee_type)
}
```

### 2. Do Not Allocate Storage for Ptr-typed Locals

The key behavioral change: a local whose type is `Ptr(T)` is already a pointer value. It does not need its own `OpVariable`. It just needs a binding in the environment from the local to an existing SPIR-V ID.

Current `Expr::Let` lowering (lines 1360-1377):
```rust
Expr::Let { local, rhs, body: let_body } => {
    let name = &body.get_local(*local).name;
    if name == "_" {
        let _ = lower_expr(constructor, body, *rhs)?;
        lower_expr(constructor, body, *let_body)
    } else {
        let value_id = lower_expr(constructor, body, *rhs)?;
        constructor.env.insert(name.clone(), value_id);
        let result = lower_expr(constructor, body, *let_body)?;
        constructor.env.remove(name);
        Ok(result)
    }
}
```

This already just binds `value_id` directly without allocating storage. The problem must be elsewhere - likely in how `Expr::Materialize` or array literals are being typed.

### Investigation Needed

Check if the array literal expressions (e5, e10, e15, e16) are incorrectly being assigned `Ptr` types instead of value types. The `OpCompositeConstruct` instructions in the disassembly are using pointer types as their result type:

```spirv
%23 = OpCompositeConstruct %_ptr_Function__arr_v4float_int_3 %float_n1 %float_n1 %float_0 %float_1
```

This suggests the type stored for the array literal expression in MIR is `Ptr(Array(...))` instead of `Array(...)`.

### Expected Result

After fix:
- Only one `OpVariable` exists for the materialized array
- Its type is a single pointer: `OpTypePointer(Function, %arr_ty)`
- No pointer-to-pointer anywhere

```spirv
%arr_ty = OpTypeArray %v4float %int_3
%ptr_arr_func = OpTypePointer Function %arr_ty

%arr_var = OpVariable %ptr_arr_func Function  ; from Materialize

; later, use %arr_var as the base pointer directly
%elem_ptr = OpAccessChain %ptr_v4float_func %arr_var %index
```
