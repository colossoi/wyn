"""CPU-only workload audit of Fold's generated code; never rewrites shader code.

The earlier experimental array-aliasing transform has been removed. The updated
compiler is responsible for evaluation placement and storage reuse.
"""
import re
FAMILIES={
 ('vec4<f32>','2483'):'positions',
 ('f32','13880'):'lambdas',
 ('vec3<f32>','9'):'grab_lambdas',
 ('i32','8192'):'heads',
 ('i32','2483'):'next',
 ('vec3<i32>','2483'):'cells',
}
ARRAY=r'array<\s*(vec[234]<\w+>|\w+)\s*,\s*(\d+)\s*>'

def audit_hash_construction(code):
    """Reject repeated hash construction inside particle/neighbor traversal.

    The source has one outer XPBD iteration loop. Hash initialization and
    insertion each add one loop. A write nested in more than two loops means
    the hash constructor was sunk into a query, as in the failed GPU attempt.
    Only structural WGSL tokens matter; comments are stripped first.
    """
    code = re.sub(r'//[^\n]*|/\*.*?\*/', '', code, flags=re.S)
    stack = []
    pending_loop = False
    writes = 0
    tokens = re.finditer(r'\bloop\b|[{}]|_scratch\.(heads|next|cells)\[[^\]\n]+\]\s*=(?!=)', code)
    for token in tokens:
        text = token.group()
        if text == 'loop':
            pending_loop = True
        elif text == '{':
            stack.append(pending_loop)
            pending_loop = False
        elif text == '}':
            if not stack:
                raise ValueError('Unbalanced WGSL braces')
            stack.pop()
        else:
            writes += 1
            if sum(stack) > 2:
                raise ValueError('Unsafe contact hash construction inside a particle/neighbor loop')
    if stack or writes < 6:
        raise ValueError('Unrecognized solver structure; refusing to publish generated code')
