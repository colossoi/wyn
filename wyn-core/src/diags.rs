//! Diagnostic utilities for AST and MIR formatting and display.
//!
//! Provides less verbose formatters for AST and MIR nodes that output
//! something close to Wyn syntax.

use crate::ast::*;
use crate::types::TypeExt;
use polytype::Type as PolyType;
use std::fmt::Write;

/// Pretty-print a polytype Type to a human-readable string.
///
/// Converts `Constructed(Str("f32"), [])` to `"f32"`,
/// `Constructed(Array, [Size(3), Str("f32")])` to `"[3]f32"`, etc.
pub fn format_type(ty: &PolyType<TypeName>) -> String {
    // Handle unique types first via dedicated API
    if let Some(inner) = ty.as_unique_inner() {
        return format!("*{}", format_type(inner));
    }
    match ty {
        PolyType::Variable(id) => format!("?{}", id),
        PolyType::Constructed(name, args) => format_constructed_type(name, args),
    }
}

fn format_constructed_type(name: &TypeName, args: &[PolyType<TypeName>]) -> String {
    match name {
        TypeName::Str(s) => {
            if args.is_empty() {
                s.to_string()
            } else {
                // Generic type application: T<A, B>
                let args_str: Vec<_> = args.iter().map(format_type).collect();
                format!("{}<{}>", s, args_str.join(", "))
            }
        }
        TypeName::Arrow => {
            // T1 -> T2
            if args.len() == 2 {
                let param = format_type(&args[0]);
                let ret = format_type(&args[1]);
                format!("{} -> {}", param, ret)
            } else if args.is_empty() {
                "() -> ?".to_string()
            } else {
                let params: Vec<_> = args[..args.len() - 1].iter().map(format_type).collect();
                let ret = format_type(&args[args.len() - 1]);
                format!("({}) -> {}", params.join(", "), ret)
            }
        }
        TypeName::Float(bits) => format!("f{}", bits),
        TypeName::UInt(bits) => format!("u{}", bits),
        TypeName::Int(bits) => format!("i{}", bits),
        TypeName::Size(n) => format!("{}", n),
        TypeName::SizeVar(s) => s.clone(),
        TypeName::SizePlaceholder => "?".to_string(),
        TypeName::Array => {
            // Array[elem, addrspace, size] - unified array type
            assert!(args.len() == 3);
            let elem = format_type(&args[0]);
            let size = format_type(&args[2]);
            // Show size and elem, omit addrspace for brevity in common case
            format!("[{}]{}", size, elem)
        }
        TypeName::Vec => {
            // vec<size>elem_type
            if args.len() == 2 {
                let size = format_type(&args[0]);
                let elem = format_type(&args[1]);
                format!("vec{}{}", size, elem)
            } else {
                "vec?".to_string()
            }
        }
        TypeName::Mat => {
            // mat<rows x cols>elem
            // Args are typically [rows, cols, elem_type]
            if args.len() == 3 {
                let rows = format_type(&args[0]);
                let cols = format_type(&args[1]);
                let elem = format_type(&args[2]);
                format!("mat{}x{}{}", rows, cols, elem)
            } else {
                "mat?".to_string()
            }
        }
        TypeName::Record(fields) => {
            // {field1: T1, field2: T2}
            // Field names are in RecordFields, field types are in args
            let items: Vec<_> = fields
                .iter()
                .zip(args.iter())
                .map(|(name, ty)| format!("{}: {}", name, format_type(ty)))
                .collect();
            format!("{{{}}}", items.join(", "))
        }
        TypeName::Unit => {
            // Unit type - ()
            "()".to_string()
        }
        TypeName::Tuple(_n) => {
            // (T1, T2, ...)
            // Tuple arity is in n, field types are in args
            let items: Vec<_> = args.iter().map(format_type).collect();
            format!("({})", items.join(", "))
        }
        TypeName::Sum(variants) => {
            // Variant1 T1 | Variant2 T2
            let items: Vec<_> = variants
                .iter()
                .map(|(name, variant_args)| {
                    if variant_args.is_empty() {
                        name.clone()
                    } else {
                        let args_str: Vec<_> = variant_args.iter().map(format_type).collect();
                        format!("{} {}", name, args_str.join(" "))
                    }
                })
                .collect();
            items.join(" | ")
        }
        TypeName::Unique => {
            // Handled in format_type() via TypeExt
            unreachable!("Unique types should be handled in format_type")
        }
        TypeName::UserVar(s) => s.clone(),
        TypeName::Named(s) => {
            if args.is_empty() {
                s.clone()
            } else {
                let args_str: Vec<_> = args.iter().map(format_type).collect();
                format!("{}<{}>", s, args_str.join(", "))
            }
        }
        TypeName::Existential(vars) => {
            // Inner type is in args[0]
            let inner = if !args.is_empty() { format_type(&args[0]) } else { "?".to_string() };
            format!("?[{}]. {}", vars.join(", "), inner)
        }
        TypeName::Pointer => {
            // Ptr<T>
            if args.len() == 1 { format!("Ptr<{}>", format_type(&args[0])) } else { "Ptr<?>".to_string() }
        }
        TypeName::ArrayVariantView => "view".to_string(),
        TypeName::ArrayVariantComposite => "composite".to_string(),
        TypeName::ArrayVariantVirtual => "virtual".to_string(),
        TypeName::AddressPlaceholder => "?variant".to_string(),
        TypeName::Skolem(id) => format!("{}", id),
        TypeName::Ignored => "_".to_string(),
    }
}

/// Formatter for AST nodes that produces readable output with line numbers.
pub struct AstFormatter {
    output: String,
    indent: usize,
    show_node_ids: bool,
}

impl AstFormatter {
    pub fn new() -> Self {
        AstFormatter {
            output: String::new(),
            indent: 0,
            show_node_ids: false,
        }
    }

    pub fn with_node_ids() -> Self {
        AstFormatter {
            output: String::new(),
            indent: 0,
            show_node_ids: true,
        }
    }

    /// Format an expression and return the formatted string.
    pub fn format_expression(expr: &Expression) -> String {
        let mut formatter = AstFormatter::new();
        formatter.write_expression(expr);
        formatter.output
    }

    /// Format a program and return the formatted string.
    pub fn format_program(program: &Program) -> String {
        let mut formatter = AstFormatter::new();
        for decl in &program.declarations {
            formatter.write_declaration(decl);
            formatter.newline();
        }
        formatter.output
    }

    /// Format a program with node IDs and return the formatted string.
    pub fn format_program_with_ids(program: &Program) -> String {
        let mut formatter = AstFormatter::with_node_ids();
        for decl in &program.declarations {
            formatter.write_declaration(decl);
            formatter.newline();
        }
        formatter.output
    }

    fn write_line(&mut self, content: &str) {
        let indent = "  ".repeat(self.indent);
        let _ = writeln!(self.output, "{}{}", indent, content);
    }

    fn newline(&mut self) {
        let _ = writeln!(self.output);
    }

    fn write_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Decl(d) => self.write_decl(d),
            Declaration::Entry(e) => self.write_entry(e),
            Declaration::Uniform(u) => {
                self.write_line(&format!("uniform {}: {}", u.name, u.ty));
            }
            Declaration::Storage(s) => {
                self.write_line(&format!("storage {}: {}", s.name, s.ty));
            }
            Declaration::Sig(v) => {
                let mut header = format!("sig {}", v.name);
                // Rust-style generics: <[n], [m], A, B>
                if !v.size_params.is_empty() || !v.type_params.is_empty() {
                    header.push('<');
                    let mut parts = Vec::new();
                    for s in &v.size_params {
                        parts.push(format!("[{}]", s));
                    }
                    for t in &v.type_params {
                        parts.push(t.clone());
                    }
                    header.push_str(&parts.join(", "));
                    header.push('>');
                }
                header.push_str(&format!(": {}", v.ty));
                self.write_line(&header);
            }
            Declaration::TypeBind(tb) => {
                self.write_line(&format!("type {} = {}", tb.name, tb.definition));
            }
            Declaration::Module(md) => {
                let name = match md {
                    ModuleDecl::Module { name, .. } => name,
                    ModuleDecl::Functor { name, .. } => name,
                };
                self.write_line(&format!("module {} = ...", name));
            }
            Declaration::ModuleTypeBind(mtb) => {
                self.write_line(&format!("module type {} = ...", mtb.name));
            }
            Declaration::Open(_) => {
                self.write_line("open ...");
            }
            Declaration::Import(path) => {
                self.write_line(&format!("import \"{}\"", path));
            }
            Declaration::Extern(e) => {
                self.write_line(&format!(
                    "#[linked(\"{}\")]\nextern {}: {}",
                    e.linkage_name, e.name, e.ty
                ));
            }
        }
    }

    fn write_decl(&mut self, decl: &Decl) {
        let mut header = format!("{} {}", decl.keyword, decl.name);

        // Rust-style generics: <[n], [m], A, B>
        if !decl.size_params.is_empty() || !decl.type_params.is_empty() {
            header.push('<');
            let mut parts = Vec::new();
            for s in &decl.size_params {
                parts.push(format!("[{}]", s));
            }
            for t in &decl.type_params {
                parts.push(t.clone());
            }
            header.push_str(&parts.join(", "));
            header.push('>');
        }

        // Rust-style comma-separated params: (x: T, y: U)
        if !decl.params.is_empty() {
            let params: Vec<String> = decl.params.iter().map(|p| self.format_pattern(p)).collect();
            header.push_str(&format!("({})", params.join(", ")));
        } else {
            header.push_str("()");
        }

        // Rust-style return type: -> T
        if let Some(ty) = &decl.ty {
            header.push_str(&format!(" -> {}", ty));
        }

        header.push_str(" =");
        self.write_line(&header);

        self.indent += 1;
        self.write_expression(&decl.body);
        self.indent -= 1;
    }

    /// Format a function parameter - bare name or (pattern: type)
    fn format_param(&self, pattern: &Pattern) -> String {
        match &pattern.kind {
            PatternKind::Name(name) => name.clone(),
            PatternKind::Typed(inner, ty) => {
                format!("({}: {})", self.format_pattern(inner), ty)
            }
            _ => format!("({})", self.format_pattern(pattern)),
        }
    }

    fn write_entry(&mut self, entry: &EntryDecl) {
        let entry_kind = if entry.entry_type.is_vertex() { "vertex" } else { "fragment" };
        let mut header = format!("{} {}", entry_kind, entry.name);

        for param in &entry.params {
            header.push_str(&format!(" {}", self.format_param(param)));
        }

        header.push_str(" =");
        self.write_line(&header);

        self.indent += 1;
        self.write_expression(&entry.body);
        self.indent -= 1;
    }

    fn write_expression(&mut self, expr: &Expression) {
        if self.show_node_ids {
            let _ = write!(self.output, "/* #{} */ ", expr.h.id.0);
        }
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                self.write_line(&n.to_string());
            }
            ExprKind::FloatLiteral(f) => {
                self.write_line(&format!("{}", f));
            }
            ExprKind::BoolLiteral(b) => {
                self.write_line(&b.to_string());
            }
            ExprKind::StringLiteral(s) => {
                self.write_line(&format!("{:?}", s));
            }
            ExprKind::Unit => {
                self.write_line("()");
            }
            ExprKind::Identifier(quals, name) => {
                let qn =
                    if quals.is_empty() { name.clone() } else { format!("{}.{}", quals.join("."), name) };
                self.write_line(&qn);
            }
            ExprKind::ArrayLiteral(elems) => {
                if elems.is_empty() {
                    self.write_line("[]");
                } else if elems.len() <= 4 && elems.iter().all(|e| self.is_simple_expr(e)) {
                    let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                    self.write_line(&format!("[{}]", items.join(", ")));
                } else {
                    self.write_line("[");
                    self.indent += 1;
                    for elem in elems {
                        self.write_expression(elem);
                    }
                    self.indent -= 1;
                    self.write_line("]");
                }
            }
            ExprKind::VecMatLiteral(elems) => {
                if elems.is_empty() {
                    self.write_line("@[]");
                } else if elems.len() <= 4 && elems.iter().all(|e| self.is_simple_expr(e)) {
                    let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                    self.write_line(&format!("@[{}]", items.join(", ")));
                } else {
                    self.write_line("@[");
                    self.indent += 1;
                    for elem in elems {
                        self.write_expression(elem);
                    }
                    self.indent -= 1;
                    self.write_line("]");
                }
            }
            ExprKind::ArrayIndex(arr, idx) => {
                let arr_str = self.format_simple_expr(arr);
                let idx_str = self.format_simple_expr(idx);
                self.write_line(&format!("{}[{}]", arr_str, idx_str));
            }
            ExprKind::ArrayWith { array, index, value } => {
                let arr_str = self.format_simple_expr(array);
                let idx_str = self.format_simple_expr(index);
                let val_str = self.format_simple_expr(value);
                self.write_line(&format!("{} with [{}] = {}", arr_str, idx_str, val_str));
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                let lhs_str = self.format_simple_expr(lhs);
                let rhs_str = self.format_simple_expr(rhs);
                self.write_line(&format!("{} {} {}", lhs_str, op.op, rhs_str));
            }
            ExprKind::UnaryOp(op, operand) => {
                let operand_str = self.format_simple_expr(operand);
                self.write_line(&format!("{}{}", op.op, operand_str));
            }
            ExprKind::Tuple(elems) => {
                let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                self.write_line(&format!("({})", items.join(", ")));
            }
            ExprKind::RecordLiteral(fields) => {
                let items: Vec<String> = fields
                    .iter()
                    .map(|(name, val)| format!("{}: {}", name, self.format_simple_expr(val)))
                    .collect();
                self.write_line(&format!("{{{}}}", items.join(", ")));
            }
            ExprKind::Lambda(lambda) => {
                let params: Vec<String> = lambda.params.iter().map(|p| self.format_pattern(p)).collect();
                self.write_line(&format!("|{}|", params.join(", ")));
                self.indent += 1;
                self.write_expression(&lambda.body);
                self.indent -= 1;
            }
            ExprKind::Application(func, args) => {
                let func_str = self.format_simple_expr(func);
                let args_str: Vec<String> = args.iter().map(|a| self.format_simple_expr(a)).collect();
                self.write_line(&format!("{} {}", func_str, args_str.join(" ")));
            }
            ExprKind::LetIn(let_in) => {
                let pat = self.format_pattern(&let_in.pattern);
                let ty = let_in.ty.as_ref().map(|t| format!(": {}", t)).unwrap_or_default();

                if self.is_simple_expr(&let_in.value) {
                    let val = self.format_simple_expr(&let_in.value);
                    self.write_line(&format!("let {}{} = {} in", pat, ty, val));
                } else {
                    self.write_line(&format!("let {}{} =", pat, ty));
                    self.indent += 1;
                    self.write_expression(&let_in.value);
                    self.indent -= 1;
                    self.write_line("in");
                }
                self.write_expression(&let_in.body);
            }
            ExprKind::FieldAccess(obj, field) => {
                let obj_str = self.format_simple_expr(obj);
                self.write_line(&format!("{}.{}", obj_str, field));
            }
            ExprKind::If(if_expr) => {
                let cond = self.format_simple_expr(&if_expr.condition);
                self.write_line(&format!("if {} then", cond));
                self.indent += 1;
                self.write_expression(&if_expr.then_branch);
                self.indent -= 1;
                self.write_line("else");
                self.indent += 1;
                self.write_expression(&if_expr.else_branch);
                self.indent -= 1;
            }
            ExprKind::Loop(loop_expr) => {
                let pat = self.format_pattern(&loop_expr.pattern);
                let init = loop_expr
                    .init
                    .as_ref()
                    .map(|e| format!(" = {}", self.format_simple_expr(e)))
                    .unwrap_or_default();
                let form = match &loop_expr.form {
                    LoopForm::For(var, bound) => {
                        format!("for {} < {}", var, self.format_simple_expr(bound))
                    }
                    LoopForm::ForIn(pat, iter) => format!(
                        "for {} in {}",
                        self.format_pattern(pat),
                        self.format_simple_expr(iter)
                    ),
                    LoopForm::While(cond) => format!("while {}", self.format_simple_expr(cond)),
                };
                self.write_line(&format!("loop {}{} {} do", pat, init, form));
                self.indent += 1;
                self.write_expression(&loop_expr.body);
                self.indent -= 1;
            }
            ExprKind::Match(match_expr) => {
                let scrut = self.format_simple_expr(&match_expr.scrutinee);
                self.write_line(&format!("match {}", scrut));
                self.indent += 1;
                for case in &match_expr.cases {
                    let pat = self.format_pattern(&case.pattern);
                    self.write_line(&format!("| {} ->", pat));
                    self.indent += 1;
                    self.write_expression(&case.body);
                    self.indent -= 1;
                }
                self.indent -= 1;
            }
            ExprKind::Range(range) => {
                let start = self.format_simple_expr(&range.start);
                let end = self.format_simple_expr(&range.end);
                let op = match range.kind {
                    RangeKind::Inclusive => "...",
                    RangeKind::Exclusive => "..",
                    RangeKind::ExclusiveLt => "..<",
                    RangeKind::ExclusiveGt => "..>",
                };
                if let Some(step) = &range.step {
                    let step_str = self.format_simple_expr(step);
                    self.write_line(&format!("{}..{}{}{}", start, step_str, op, end));
                } else {
                    self.write_line(&format!("{}{}{}", start, op, end));
                }
            }
            ExprKind::Slice(slice) => {
                let arr = self.format_simple_expr(&slice.array);
                let start = slice.start.as_ref().map(|e| self.format_simple_expr(e)).unwrap_or_default();
                let end = slice.end.as_ref().map(|e| self.format_simple_expr(e)).unwrap_or_default();
                self.write_line(&format!("{}[{}:{}]", arr, start, end));
            }
            ExprKind::TypeAscription(inner, ty) => {
                let inner_str = self.format_simple_expr(inner);
                self.write_line(&format!("{} : {}", inner_str, ty));
            }
            ExprKind::TypeCoercion(inner, ty) => {
                let inner_str = self.format_simple_expr(inner);
                self.write_line(&format!("{} :> {}", inner_str, ty));
            }
            ExprKind::TypeHole => {
                self.write_line("???");
            }
        }
    }

    fn is_simple_expr(&self, expr: &Expression) -> bool {
        matches!(
            &expr.kind,
            ExprKind::IntLiteral(_)
                | ExprKind::FloatLiteral(_)
                | ExprKind::BoolLiteral(_)
                | ExprKind::StringLiteral(_)
                | ExprKind::Unit
                | ExprKind::Identifier(_, _)
                | ExprKind::TypeHole
        )
    }

    fn format_simple_expr(&self, expr: &Expression) -> String {
        match &expr.kind {
            ExprKind::IntLiteral(n) => n.to_string(),
            ExprKind::FloatLiteral(f) => format!("{}", f),
            ExprKind::BoolLiteral(b) => b.to_string(),
            ExprKind::StringLiteral(s) => format!("{:?}", s),
            ExprKind::Unit => "()".to_string(),
            ExprKind::Identifier(quals, name) => {
                if quals.is_empty() {
                    name.clone()
                } else {
                    format!("{}.{}", quals.join("."), name)
                }
            }
            ExprKind::TypeHole => "???".to_string(),
            ExprKind::Tuple(elems) => {
                let items: Vec<String> = elems.iter().map(|e| self.format_simple_expr(e)).collect();
                format!("({})", items.join(", "))
            }
            ExprKind::BinaryOp(op, lhs, rhs) => {
                format!(
                    "({} {} {})",
                    self.format_simple_expr(lhs),
                    op.op,
                    self.format_simple_expr(rhs)
                )
            }
            ExprKind::UnaryOp(op, operand) => {
                format!("({}{})", op.op, self.format_simple_expr(operand))
            }
            ExprKind::Application(func, args) => {
                let func_str = self.format_simple_expr(func);
                let args_str: Vec<String> = args.iter().map(|a| self.format_simple_expr(a)).collect();
                format!("({} {})", func_str, args_str.join(" "))
            }
            ExprKind::ArrayIndex(arr, idx) => {
                format!(
                    "{}[{}]",
                    self.format_simple_expr(arr),
                    self.format_simple_expr(idx)
                )
            }
            ExprKind::ArrayWith { array, index, value } => {
                format!(
                    "{} with [{}] = {}",
                    self.format_simple_expr(array),
                    self.format_simple_expr(index),
                    self.format_simple_expr(value)
                )
            }
            ExprKind::FieldAccess(obj, field) => {
                format!("{}.{}", self.format_simple_expr(obj), field)
            }
            ExprKind::TypeAscription(inner, ty) => {
                format!("({}: {})", self.format_simple_expr(inner), ty)
            }
            _ => "<complex>".to_string(),
        }
    }

    fn format_pattern(&self, pattern: &Pattern) -> String {
        match &pattern.kind {
            PatternKind::Name(name) => name.clone(),
            PatternKind::Wildcard => "_".to_string(),
            PatternKind::Unit => "()".to_string(),
            PatternKind::Literal(lit) => match lit {
                PatternLiteral::Int(n) => n.to_string(),
                PatternLiteral::Float(f) => format!("{}", f),
                PatternLiteral::Char(c) => format!("'{}'", c),
                PatternLiteral::Bool(b) => b.to_string(),
            },
            PatternKind::Tuple(patterns) => {
                let items: Vec<String> = patterns.iter().map(|p| self.format_pattern(p)).collect();
                format!("({})", items.join(", "))
            }
            PatternKind::Record(fields) => {
                let items: Vec<String> = fields
                    .iter()
                    .map(|f| {
                        if let Some(pat) = &f.pattern {
                            format!("{} = {}", f.field, self.format_pattern(pat))
                        } else {
                            f.field.clone()
                        }
                    })
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            PatternKind::Constructor(name, patterns) => {
                if patterns.is_empty() {
                    name.clone()
                } else {
                    let items: Vec<String> = patterns.iter().map(|p| self.format_pattern(p)).collect();
                    format!("{} {}", name, items.join(" "))
                }
            }
            PatternKind::Typed(inner, ty) => {
                format!("{}: {}", self.format_pattern(inner), ty)
            }
            PatternKind::Attributed(attrs, inner) => {
                let attr_str = attrs.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>().join(" ");
                format!("#[{}] {}", attr_str, self.format_pattern(inner))
            }
        }
    }
}

impl Default for AstFormatter {
    fn default() -> Self {
        Self::new()
    }
}

// MIR Display implementations
use crate::mir;
use std::fmt::{self, Display, Formatter};

impl Display for mir::Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, def) in self.defs.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", def)?;
        }
        Ok(())
    }
}

impl Display for mir::Def {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::Def::Function {
                name,
                params,
                ret_type,
                attributes,
                body,
                ..
            } => {
                // Write attributes
                for attr in attributes {
                    writeln!(f, "{}", attr)?;
                }
                // Write function signature with types
                write!(f, "def {}", name)?;
                for param_id in params.iter() {
                    let param = body.get_local(*param_id);
                    write!(f, " ({}: {})", param.name, format_type(&param.ty))?;
                }
                write!(f, ": {}", format_type(ret_type))?;
                writeln!(f, " =")?;
                // Write body with indentation
                write!(f, "  {}", body)
            }
            mir::Def::Constant {
                name,
                ty,
                attributes,
                body,
                ..
            } => {
                // Write attributes
                for attr in attributes {
                    writeln!(f, "{}", attr)?;
                }
                // Write constant with type
                writeln!(f, "def {}: {} =", name, format_type(ty))?;
                write!(f, "  {}", body)
            }
            mir::Def::Uniform {
                name,
                ty,
                set,
                binding,
                ..
            } => {
                write!(
                    f,
                    "#[uniform(set={}, binding={})] def {}: {}",
                    set,
                    binding,
                    name,
                    format_type(ty)
                )
            }
            mir::Def::Storage {
                name,
                ty,
                set,
                binding,
                ..
            } => {
                write!(
                    f,
                    "#[storage(set={}, binding={})] def {}: {}",
                    set,
                    binding,
                    name,
                    format_type(ty)
                )
            }
            mir::Def::EntryPoint {
                name,
                execution_model,
                inputs,
                outputs,
                body,
                ..
            } => {
                // Write execution model
                match execution_model {
                    mir::ExecutionModel::Vertex => writeln!(f, "#[vertex]")?,
                    mir::ExecutionModel::Fragment => writeln!(f, "#[fragment]")?,
                    mir::ExecutionModel::Compute { local_size } => {
                        writeln!(
                            f,
                            "#[compute({}, {}, {})]",
                            local_size.0, local_size.1, local_size.2
                        )?;
                    }
                }
                // Write entry signature with inputs
                write!(f, "entry {}", name)?;
                for input in inputs.iter() {
                    let decoration = match &input.decoration {
                        Some(mir::IoDecoration::Location(loc)) => format!(" @location({})", loc),
                        Some(mir::IoDecoration::BuiltIn(b)) => format!(" @builtin({:?})", b),
                        None => String::new(),
                    };
                    write!(f, " ({}{}: {})", input.name, decoration, format_type(&input.ty))?;
                }
                // Write return type
                if outputs.len() == 1 {
                    let out = &outputs[0];
                    let decoration = match &out.decoration {
                        Some(mir::IoDecoration::Location(loc)) => format!(" @location({})", loc),
                        Some(mir::IoDecoration::BuiltIn(b)) => format!(" @builtin({:?})", b),
                        None => String::new(),
                    };
                    write!(f, ":{} {}", decoration, format_type(&out.ty))?;
                } else if !outputs.is_empty() {
                    write!(f, ": (")?;
                    for (i, out) in outputs.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        let decoration = match &out.decoration {
                            Some(mir::IoDecoration::Location(loc)) => format!("@location({}) ", loc),
                            Some(mir::IoDecoration::BuiltIn(b)) => format!("@builtin({:?}) ", b),
                            None => String::new(),
                        };
                        write!(f, "{}{}", decoration, format_type(&out.ty))?;
                    }
                    write!(f, ")")?;
                }
                writeln!(f, " =")?;
                // Write body with indentation
                write!(f, "  {}", body)
            }
        }
    }
}

impl Display for mir::Param {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}: {})", self.name, format_type(&self.ty))
    }
}

impl Display for mir::Attribute {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::Attribute::BuiltIn(builtin) => write!(f, "#[builtin({:?})]", builtin),
            mir::Attribute::Location(loc) => write!(f, "#[location({})]", loc),
            mir::Attribute::Vertex => write!(f, "#[vertex]"),
            mir::Attribute::Fragment => write!(f, "#[fragment]"),
            mir::Attribute::Compute => write!(f, "#[compute]"),
            mir::Attribute::Uniform => write!(f, "#[uniform]"),
            mir::Attribute::Storage => write!(f, "#[storage]"),
            mir::Attribute::SizeHint(hint) => write!(f, "#[size_hint({})]", hint),
        }
    }
}

// Display implementations for MIR types.

impl Display for mir::Body {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "  locals:")?;
        for (i, local) in self.locals.iter().enumerate() {
            writeln!(f, "    {}: {} ({})", i, local.name, local.ty)?;
        }
        writeln!(f, "  exprs:")?;
        for (i, expr) in self.exprs.iter().enumerate() {
            let marker = if mir::ExprId(i as u32) == self.root { " <-- root" } else { "" };
            writeln!(f, "    e{}: {}{}", i, expr, marker)?;
        }
        Ok(())
    }
}

impl Display for mir::Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::Expr::Local(id) => write!(f, "local_{}", id.0),
            mir::Expr::Global(name) => write!(f, "{}", name),
            mir::Expr::Extern(linkage) => write!(f, "extern \"{}\"", linkage),
            mir::Expr::Int(s) => write!(f, "{}", s),
            mir::Expr::Float(s) => write!(f, "{}", s),
            mir::Expr::Bool(b) => write!(f, "{}", b),
            mir::Expr::Unit => write!(f, "()"),
            mir::Expr::String(s) => write!(f, "\"{}\"", s.escape_default()),
            mir::Expr::Tuple(ids) => {
                write!(f, "(")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "e{}", id.0)?;
                }
                write!(f, ")")
            }
            mir::Expr::Array { backing, size } => match backing {
                mir::ArrayBacking::Literal(ids) => {
                    write!(f, "[")?;
                    for (i, id) in ids.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "e{}", id.0)?;
                    }
                    write!(f, "]")
                }
                mir::ArrayBacking::Range { start, step, kind } => {
                    let kind_str = match kind {
                        mir::RangeKind::Inclusive => "...",
                        mir::RangeKind::Exclusive => "..",
                        mir::RangeKind::ExclusiveLt => "..<",
                        mir::RangeKind::ExclusiveGt => "..>",
                    };
                    if let Some(step) = step {
                        write!(f, "e{}..e{}{}e{}", start.0, step.0, kind_str, size.0)
                    } else {
                        write!(f, "e{}{}e{}", start.0, kind_str, size.0)
                    }
                }
            },
            mir::Expr::Vector(ids) => {
                write!(f, "@[")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "e{}", id.0)?;
                }
                write!(f, "]")
            }
            mir::Expr::Matrix(rows) => {
                write!(f, "@[")?;
                for (i, row) in rows.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[")?;
                    for (j, id) in row.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "e{}", id.0)?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")
            }
            mir::Expr::BinOp { op, lhs, rhs } => {
                write!(f, "(e{} {} e{})", lhs.0, op, rhs.0)
            }
            mir::Expr::UnaryOp { op, operand } => {
                write!(f, "({}e{})", op, operand.0)
            }
            mir::Expr::Let { local, rhs, body } => {
                write!(f, "let local_{} = e{} in e{}", local.0, rhs.0, body.0)
            }
            mir::Expr::If { cond, then_, else_ } => {
                write!(f, "if e{} then e{} else e{}", cond.0, then_.0, else_.0)
            }
            mir::Expr::Loop {
                loop_var,
                init,
                kind,
                body,
                ..
            } => {
                write!(f, "loop local_{} = e{} ", loop_var.0, init.0)?;
                match kind {
                    mir::LoopKind::For { var, iter } => {
                        write!(f, "for local_{} in e{}", var.0, iter.0)?;
                    }
                    mir::LoopKind::ForRange { var, bound } => {
                        write!(f, "for local_{} < e{}", var.0, bound.0)?;
                    }
                    mir::LoopKind::While { cond } => {
                        write!(f, "while e{}", cond.0)?;
                    }
                }
                write!(f, " do e{}", body.0)
            }
            mir::Expr::Call { func, args } => {
                write!(f, "{}(", func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "e{}", arg.0)?;
                }
                write!(f, ")")
            }
            mir::Expr::Intrinsic { name, args } => {
                write!(f, "@{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "e{}", arg.0)?;
                }
                write!(f, ")")
            }
            mir::Expr::Materialize(inner) => {
                write!(f, "@materialize(e{})", inner.0)
            }
            mir::Expr::Attributed { attributes, expr } => {
                for attr in attributes {
                    write!(f, "{} ", attr)?;
                }
                write!(f, "e{}", expr.0)
            }
            mir::Expr::Load { ptr } => {
                write!(f, "@load(e{})", ptr.0)
            }
            mir::Expr::Store { ptr, value } => {
                write!(f, "@store(e{}, e{})", ptr.0, value.0)
            }
            mir::Expr::StorageView { set, binding, offset, len } => {
                write!(f, "@storage_view(set={}, binding={}, offset=e{}, len=e{})", set, binding, offset.0, len.0)
            }
            mir::Expr::SliceStorageView { view, start, len } => {
                write!(f, "@slice_storage_view(e{}, start=e{}, len=e{})", view.0, start.0, len.0)
            }
            mir::Expr::StorageViewIndex { view, index } => {
                write!(f, "@storage_view_index(e{}, e{})", view.0, index.0)
            }
            mir::Expr::StorageViewLen { view } => {
                write!(f, "@storage_view_len(e{})", view.0)
            }
        }
    }
}

impl Display for mir::LoopKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            mir::LoopKind::For { var, iter } => {
                write!(f, "for local_{} in e{}", var.0, iter.0)
            }
            mir::LoopKind::ForRange { var, bound } => {
                write!(f, "for local_{} < e{}", var.0, bound.0)
            }
            mir::LoopKind::While { cond } => {
                write!(f, "while e{}", cond.0)
            }
        }
    }
}

// =============================================================================
// TLC Display implementations
// =============================================================================

use crate::tlc;

impl Display for tlc::Program {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (i, def) in self.defs.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
                writeln!(f)?;
            }
            write!(f, "{}", def)?;
        }
        Ok(())
    }
}

impl Display for tlc::Def {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.name, self.body)
    }
}

impl Display for tlc::Term {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_prec(f, 0)
    }
}

impl tlc::Term {
    fn fmt_prec(&self, f: &mut Formatter<'_>, prec: usize) -> fmt::Result {
        match &self.kind {
            tlc::TermKind::Var(name) => write!(f, "{}", name),

            tlc::TermKind::Lam {
                param,
                param_ty,
                body,
            } => {
                if prec > 0 {
                    write!(f, "(")?;
                }
                write!(f, "λ({}: {}). ", param, format_type(param_ty))?;
                body.fmt_prec(f, 0)?;
                if prec > 0 {
                    write!(f, ")")?;
                }
                Ok(())
            }

            tlc::TermKind::App { func, arg } => {
                if prec > 1 {
                    write!(f, "(")?;
                }
                func.fmt_prec(f, 1)?;
                write!(f, " ")?;
                arg.fmt_prec(f, 2)?;
                if prec > 1 {
                    write!(f, ")")?;
                }
                Ok(())
            }

            tlc::TermKind::BinOp(op) => write!(f, "({})", op.op),
            tlc::TermKind::UnOp(op) => write!(f, "({})", op.op),

            tlc::TermKind::Let {
                name,
                name_ty,
                rhs,
                body,
            } => {
                if prec > 0 {
                    write!(f, "(")?;
                }
                write!(f, "let {}: {} = ", name, format_type(name_ty))?;
                rhs.fmt_prec(f, 0)?;
                write!(f, " in ")?;
                body.fmt_prec(f, 0)?;
                if prec > 0 {
                    write!(f, ")")?;
                }
                Ok(())
            }

            tlc::TermKind::IntLit(n) => write!(f, "{}", n),
            tlc::TermKind::FloatLit(n) => write!(f, "{}", n),
            tlc::TermKind::BoolLit(b) => write!(f, "{}", b),
            tlc::TermKind::StringLit(s) => write!(f, "\"{}\"", s),

            tlc::TermKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                if prec > 0 {
                    write!(f, "(")?;
                }
                write!(f, "if ")?;
                cond.fmt_prec(f, 0)?;
                write!(f, " then ")?;
                then_branch.fmt_prec(f, 0)?;
                write!(f, " else ")?;
                else_branch.fmt_prec(f, 0)?;
                if prec > 0 {
                    write!(f, ")")?;
                }
                Ok(())
            }

            tlc::TermKind::Loop {
                loop_var,
                init,
                kind,
                body,
                ..
            } => {
                if prec > 0 {
                    write!(f, "(")?;
                }
                write!(f, "loop {} = ", loop_var)?;
                init.fmt_prec(f, 0)?;
                match kind {
                    tlc::LoopKind::For { var, .. } => write!(f, " for {} in ... ", var)?,
                    tlc::LoopKind::ForRange { var, bound, .. } => {
                        write!(f, " for {} < ", var)?;
                        bound.fmt_prec(f, 0)?;
                    }
                    tlc::LoopKind::While { cond } => {
                        write!(f, " while ")?;
                        cond.fmt_prec(f, 0)?;
                    }
                }
                write!(f, " do ")?;
                body.fmt_prec(f, 0)?;
                if prec > 0 {
                    write!(f, ")")?;
                }
                Ok(())
            }

            tlc::TermKind::Extern(linkage_name) => {
                write!(f, "extern \"{}\"", linkage_name)
            }
        }
    }
}
