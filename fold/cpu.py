"""Adapt Naga's scalar Metal C++ output to native Clang vector operations.

Equations, control flow, and array accesses come from the compiled Wyn module.
Only platform type/builtin spellings and the buffer ABI are adapted here.
"""
import re


HEADER=r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#define device
#define constant
#define thread
#define kernel
template<class To,class From> To as_type(From value) { return __builtin_bit_cast(To,value); }
namespace metal {
using uint=unsigned int;
using float2=float __attribute__((ext_vector_type(2)));
using float3=float __attribute__((ext_vector_type(3)));
using float4=float __attribute__((ext_vector_type(4)));
using int3=int __attribute__((ext_vector_type(3)));
using uint2=uint __attribute__((ext_vector_type(2)));
using uint3=uint __attribute__((ext_vector_type(3)));
struct packed_float3 {
 float x,y,z;
 packed_float3()=default;
 packed_float3(float3 v):x(v.x),y(v.y),z(v.z){}
 operator float3() const { return float3{x,y,z}; }
};
using std::min;using std::max;using std::clamp;
using std::sqrt;using std::ceil;using std::floor;using std::pow;
template<class T> T select(T no,T yes,bool condition) {return condition?yes:no;}
template<class T> bool all(T v) {return v.x && v.y;}
inline float dot(float3 a,float3 b) {return (a.x*b.x+a.y*b.y)+a.z*b.z;}
inline float length(float3 a) {return sqrt(dot(a,a));}
}
inline metal::float2 fold_float2(float x,float y) {return {x,y};}
inline metal::float2 fold_float2(float x) {return {x,x};}
inline metal::float3 fold_float3(float x,float y,float z) {return {x,y,z};}
inline metal::float3 fold_float3(float x) {return {x,x,x};}
inline metal::float3 fold_float3(metal::float3 x) {return x;}
inline metal::float4 fold_float4(float x,float y,float z,float w) {return {x,y,z,w};}
inline metal::float4 fold_float4(float x) {return {x,x,x,x};}
inline metal::int3 fold_int3(int x,int y,int z) {return {x,y,z};}
inline metal::int3 fold_int3(int x) {return {x,x,x};}
inline metal::uint2 fold_uint2(unsigned x,unsigned y) {return {x,y};}
inline metal::uint2 fold_uint2(unsigned x) {return {x,x};}
inline metal::uint3 fold_uint3(unsigned x) {return {x,x,x};}
using metal::uint2;
'''


def native_cpp(source):
    if source.count('kernel void constraints_compute(')!=1:
        raise ValueError('Expected one serial constraints kernel')
    if 'fold_scratch' in source or 'threadgroup ' in source:
        raise ValueError('Native CPU compilation requires the original private-array kernel')
    source=re.sub(r'^#include[^\n]*\n','',source,flags=re.M)
    source=re.sub(r'\[\[[^\]]*\]\]','',source)
    # Runtime storage arrays are pointer parameters in C++, avoiding the dummy
    # one-element array type Naga uses as Metal's unbounded-array spelling.
    source=re.sub(r'type_\d+ const& (_buf_\d_\d_)',r'metal::float4 const* \1',source)
    source=re.sub(r'type_\d+& (_buf_\d_\d_)',r'metal::float4* \1',source)
    for ty in ['float2','float3','float4','int3','uint2','uint3']:
        source=source.replace('metal::'+ty+'(','fold_'+ty+'(')
    source=re.sub(r'(?<![\w:])uint2\(', 'fold_uint2(', source)
    source=re.sub(r'\b(\d+\.\d+(?:[eE][+-]?\d+)?)\b',r'\1f',source)
    # Reject an ABI change rather than silently supplying the wrong buffers.
    parameters=source[source.index('kernel void constraints_compute('):].split(') {',1)[0]
    expected=['_wgsl_gid','_wgsl_lid','_wgsl_nwg','_wgsl_group','_pc0_',
              '_buf_0_0_','_buf_0_1_','_buf_0_2_','_buf_0_3_','_buf_0_4_','_buffer_sizes']
    locations=[parameters.index(name) for name in expected]
    if locations!=sorted(locations):raise ValueError('Unexpected native kernel ABI')
    wrapper=r'''
extern "C" void fold_cpu(const float* p,const float* old,const float* edges,
                         const float* parameters,const float* grabs,float* output) {
 alignas(16) metal::float4 input[2483],previous[2483],links[13880],grab[9],result[2483];
 _PcBlock0_ config;
 std::memcpy(input,p,sizeof(input));std::memcpy(previous,old,sizeof(previous));
 std::memcpy(links,edges,sizeof(links));std::memcpy(grab,grabs,sizeof(grab));
 static_assert(sizeof(config)==176,"Parameter layout mismatch");
 std::memcpy(&config,parameters,sizeof(config));
 _mslBufferSizes sizes={};
 constraints_compute(fold_uint3(0),fold_uint3(0),fold_uint3(1),fold_uint3(0),
                     config,input,previous,links,grab,result,sizes);
 std::memcpy(output,result,sizeof(result));
}
'''
    return HEADER+source+wrapper
