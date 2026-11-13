//! Performance workaround: egglog 3.0's primitive macro clones container inputs,
//! even for read-only operations. PurePrim receives raw Value handles; borrowing
//! here avoids quadratic copying during repeated lookups/length checks.
//! Restore upstream reads when they borrow containers without cloning them.
use egglog_engine::ast::Span;
use egglog_engine::constraint::{SimpleTypeConstraint, TypeConstraint};
use egglog_engine::prelude::BaseSort;
use egglog_engine::sort::{I64Sort, MapContainer, SetContainer, UnitSort, VecContainer};
use egglog_engine::{ArcSort, Core, EGraph, Primitive, PurePrim, PureState, Value};
use std::any::TypeId;

pub(super) fn register(graph: &mut EGraph) {
    for sort in graph.get_arcsorts_by(|sort| {
        [
            TypeId::of::<VecContainer>(),
            TypeId::of::<MapContainer>(),
            TypeId::of::<SetContainer>(),
        ]
        .into_iter()
        .any(|id| sort.value_type() == Some(id))
    }) {
        let names: &[&str] = if sort.value_type() == Some(TypeId::of::<VecContainer>()) {
            &["vec-at", "vec-size"]
        } else if sort.value_type() == Some(TypeId::of::<MapContainer>()) {
            &["map-at", "map-size", "map-includes", "map-excludes"]
        } else {
            &["set-at", "set-size", "set-includes", "set-excludes"]
        };
        for &name in names {
            graph.add_pure_primitive(BorrowedRead(sort.clone(), name), None);
        }
    }
}

// Egglog clones this sort metadata during registration, not runtime vector contents.
#[derive(Clone)]
struct BorrowedRead(ArcSort, &'static str);

impl Primitive for BorrowedRead {
    fn name(&self) -> &str {
        self.1
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        let inner = self.0.inner_sorts();
        let sorts = match self.1 {
            "vec-at" | "set-at" => vec![self.0.clone(), I64Sort.to_arcsort(), inner[0].clone()],
            "map-at" => vec![self.0.clone(), inner[0].clone(), inner[1].clone()],
            name if name.ends_with("-size") => vec![self.0.clone(), I64Sort.to_arcsort()],
            _ => vec![self.0.clone(), inner[0].clone(), UnitSort.to_arcsort()],
        };
        SimpleTypeConstraint::new(self.name(), sorts, span.clone()).into_box()
    }
}

impl PurePrim for BorrowedRead {
    #[inline(always)]
    fn apply<'a, 'db>(&self, state: PureState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let container = *args.first()?;
        let index = || usize::try_from(state.base_values().unwrap::<i64>(*args.get(1)?)).ok();
        let values = state.container_values();
        let size = match self.1 {
            "vec-at" => return values.get_val::<VecContainer>(container)?.data.get(index()?).copied(),
            "map-at" => return values.get_val::<MapContainer>(container)?.data.get(args.get(1)?).copied(),
            "set-at" => {
                return values.get_val::<SetContainer>(container)?.data.iter().nth(index()?).copied()
            }
            "vec-size" => values.get_val::<VecContainer>(container)?.data.len(),
            "map-size" => values.get_val::<MapContainer>(container)?.data.len(),
            "set-size" => values.get_val::<SetContainer>(container)?.data.len(),
            name => {
                let key = args.get(1)?;
                let present = if name.starts_with("map-") {
                    values.get_val::<MapContainer>(container)?.data.contains_key(key)
                } else {
                    values.get_val::<SetContainer>(container)?.data.contains(key)
                };
                return (present == name.ends_with("-includes")).then(|| state.base_values().get(()));
            }
        };
        Some(state.base_values().get(size as i64))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn reads_preserve_lengths_order_missing_keys_and_guard_semantics() {
        let mut graph = egglog_engine::EGraph::default();
        graph.parse_and_run_program(None, "(datatype Key (KeyId i64)) (sort Keys (Vec Key)) (sort KeySet (Set Key)) (sort KeyMap (Map Key Key)) (sort Routes (Map Key Keys))").unwrap();
        super::register(&mut graph);
        graph
            .parse_and_run_program(
                None,
                r#"
            (let a (KeyId 0)) (let b (KeyId 1)) (let c (KeyId 2))
            (let v (vec-of a b)) (let s (set-of a b))
            (let m (map-insert (map-empty) a b))
            (let routes (map-insert (map-empty) a v))
            (check (= (vec-at v 1) (vec-get v 1)))
            (check (= (map-at m a) (map-get m a)))
            (check (= (map-at routes a) v))
            (check (= (set-at s 0) (set-get s 0)))
            (check (= (vec-size v) (vec-length v)))
            (check (= (map-size m) (map-length m)))
            (check (= (set-size s) (set-length s)))
            (check (= (vec-size (vec-remove (vec-remove v 0) 0)) 0))
            (check (= (map-size (map-remove m a)) 0))
            (check (= (set-size (set-diff s s)) 0))
            (check (map-includes m a)) (check (map-excludes m c))
            (check (set-includes s a)) (check (set-excludes s c))
            (fail (check (map-includes m c))) (fail (check (map-excludes m a)))
            (fail (check (set-includes s c))) (fail (check (set-excludes s a)))
            (fail (check (= (vec-at v -1) a))) (fail (check (= (vec-at v 2) a)))
            (fail (check (= (set-at s -1) a))) (fail (check (= (set-at s 2) a)))
            (fail (check (= (map-at m c) a)))
        "#,
            )
            .unwrap();
    }
}
