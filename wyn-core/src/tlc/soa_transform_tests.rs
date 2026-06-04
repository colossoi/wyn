use crate::tlc::soa::*;

fn i32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Int(32), vec![])
}

fn f32_ty() -> Type<TypeName> {
    Type::Constructed(TypeName::Float(32), vec![])
}

fn size_ty(n: usize) -> Type<TypeName> {
    Type::Constructed(TypeName::Size(n), vec![])
}

fn composite_variant() -> Type<TypeName> {
    Type::Constructed(TypeName::ArrayVariantComposite, vec![])
}

fn array_ty(elem: Type<TypeName>, size: usize) -> Type<TypeName> {
    Type::Constructed(TypeName::Array, vec![elem, composite_variant(), size_ty(size)])
}

fn tuple_ty(args: Vec<Type<TypeName>>) -> Type<TypeName> {
    Type::Constructed(TypeName::Tuple(args.len()), args)
}

#[test]
fn test_soa_type_scalar() {
    assert_eq!(soa_type(&i32_ty()), i32_ty());
    assert_eq!(soa_type(&f32_ty()), f32_ty());
}

#[test]
fn test_soa_type_plain_array() {
    let arr = array_ty(f32_ty(), 4);
    assert_eq!(soa_type(&arr), arr);
}

#[test]
fn test_soa_type_array_of_tuple() {
    // [4](i32, f32) → ([4]i32, [4]f32)
    let arr = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
    let expected = tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]);
    assert_eq!(soa_type(&arr), expected);
}

#[test]
fn test_soa_type_nested_array() {
    // [n][m](A,B) → ([n][m]A, [n][m]B)
    let inner = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 3);
    let outer = array_ty(inner, 5);
    let result = soa_type(&outer);

    // First soa_type on outer: elem is [3](i32,f32), which soa_type transforms to ([3]i32, [3]f32)
    // That's a tuple, so the outer array distributes:
    // ([5]([3]i32), [5]([3]f32)) — but wait, [5]([3]i32) is [5][3]i32 which is fine (no tuple elem)
    // Actually: soa_type([5][3](i32,f32)):
    //   elem = [3](i32,f32), soa_type → ([3]i32, [3]f32), which is Tuple
    //   So outer distributes: ([5]([3]i32), [5]([3]f32))
    //   But [5]([3]i32) has elem = ([3]i32) which is NOT a tuple, so it stays.
    // Actually ([3]i32) is Array, not Tuple. So the elem of the outer after soa
    // is ([3]i32, [3]f32) which IS a tuple. So we get:
    // (Array[([3]i32), composite, 5], Array[([3]f32), composite, 5])
    // = ([5][3]i32 via nesting... no, it's [5]([3]i32))
    // Hmm, [5](something) where something = Tuple. So distribute:
    // The soa_type of the inner element is ([3]i32, [3]f32).
    // Distributing array over this tuple: ([5]([3]i32), [5]([3]f32))
    // These are arrays whose elements are arrays (not tuples), so no further transformation.
    let expected = tuple_ty(vec![
        array_ty(array_ty(i32_ty(), 3), 5),
        array_ty(array_ty(f32_ty(), 3), 5),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_soa_type_standalone_tuple() {
    // (A, [n](B,C)) → (A, ([n]B, [n]C))
    let inner = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
    let standalone = tuple_ty(vec![f32_ty(), inner]);
    let result = soa_type(&standalone);
    let expected = tuple_ty(vec![
        f32_ty(),
        tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]),
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_soa_type_arrow() {
    // ([4](i32,f32)) -> i32  →  ([4]i32, [4]f32) -> i32
    let param = array_ty(tuple_ty(vec![i32_ty(), f32_ty()]), 4);
    let arrow = Type::Constructed(TypeName::Arrow, vec![param, i32_ty()]);
    let result = soa_type(&arrow);
    let expected = Type::Constructed(
        TypeName::Arrow,
        vec![
            tuple_ty(vec![array_ty(i32_ty(), 4), array_ty(f32_ty(), 4)]),
            i32_ty(),
        ],
    );
    assert_eq!(result, expected);
}
