// Native rendering of the upstream shell/material algorithms. No browser runtime.
struct Frame {
 right:vec4f, up:vec4f, forward:vec4f, eye:vec4f,
 viewport:vec4f, optics:vec4f, grade:vec4f, switches:vec4f,
}
@group(0) @binding(0) var<uniform> frame:Frame;
@group(0) @binding(1) var front_depth:texture_depth_2d;
@group(0) @binding(2) var behind_color:texture_2d<f32>;
@group(0) @binding(3) var behind_depth:texture_depth_2d;
@group(0) @binding(4) var image:texture_2d<f32>;
@group(0) @binding(5) var normals:texture_2d<f32>;
@group(0) @binding(6) var linear_sampler:sampler;
@group(0) @binding(7) var ambient_texture:texture_2d<f32>;
@group(0) @binding(8) var graded_texture:texture_2d<f32>;
struct Surface {
 @builtin(position) clip:vec4f,
 @location(0) world:vec3f, @location(1) normal:vec3f,
 @location(2) birth:f32,@location(3) fade:f32,
}
fn view(p:vec3f)->vec3f { let q=p-frame.eye.xyz;return vec3f(dot(q,frame.right.xyz),dot(q,frame.up.xyz),-dot(q,frame.forward.xyz)); }
fn world_direction(p:vec3f)->vec3f {return frame.right.xyz*p.x+frame.up.xyz*p.y-frame.forward.xyz*p.z;}
fn project(p:vec3f)->vec2f {return vec2f(p.x*frame.viewport.z,-p.y*frame.viewport.w)/(-p.z)*0.5+0.5;}
fn reconstruct(uv:vec2f,depth:f32)->vec3f {let z=6.0/(60.0-depth*59.9);return vec3f((uv.x*2.0-1.0)*z/frame.viewport.z,(1.0-uv.y*2.0)*z/frame.viewport.w,-z);}
@vertex fn vertex(@location(0) p:vec3f,@location(1) n:vec3f,@location(2) birth:f32,@location(3) fade:f32)->Surface {
 let q=view(p);let z=-q.z;var out:Surface;
 out.clip=vec4f(q.x*frame.viewport.z,q.y*frame.viewport.w,z*60.0/59.9-6.0/59.9,z);
 out.world=p;out.normal=n;out.birth=birth;out.fade=fade;return out;
}
fn coverage(s:Surface) {
 if s.world.y<0.0 {discard;}
 // Spatial hashed alpha coverage, common to every depth-writing pass.
 let noise=fract(sin(dot(floor(s.world*1200.0),vec3f(12.9898,78.233,37.719)))*43758.5453);
 if noise>s.fade {discard;}
}
fn pigment(birth:f32)->vec3f {
 let h=birth*0.065+0.05;
 let rainbow=clamp(abs(fract(h+vec3f(0.0,2.0/3.0,1.0/3.0))*6.0-3.0)-1.0,vec3f(0),vec3f(1));
 let rainbow_smooth=rainbow*rainbow*(3.0-2.0*rainbow);
 return mix(vec3f(1),mix(vec3f(0.025),vec3f(0.7),rainbow_smooth),frame.optics.z);
}
fn environment(direction:vec3f,roughness:f32)->vec3f {
 // Pastel studio, warm/cool softboxes, matching the upstream room lighting.
 let d=normalize(direction);var color=mix(vec3f(0.38,0.34,0.48),vec3f(0.73,0.81,0.89),smoothstep(-0.6,0.7,d.y));
 let spread=mix(160.0,3.0,roughness*roughness);
 color+=vec3f(1.0,0.66,0.48)*pow(max(dot(d,normalize(vec3f(-1.0,0.5,0.7))),0.0),spread)*2.4;
 color+=vec3f(0.50,0.82,1.0)*pow(max(dot(d,normalize(vec3f(1.0,0.6,-0.3))),0.0),spread)*2.0;
 color+=vec3f(1.0,0.62,0.80)*pow(max(dot(d,normalize(vec3f(0.1,0.7,1.0))),0.0),spread)*1.8;
 return color;
}
fn fresnel(c:f32,f0:vec3f)->vec3f {return f0+(1.0-f0)*pow(1.0-clamp(c,0.0,1.0),5.0);}
fn brdf(n:vec3f,v:vec3f,l:vec3f,color:vec3f,r:f32,f0:vec3f)->vec3f {
 let h=normalize(v+l);let nl=max(dot(n,l),0.0);let nv=max(dot(n,v),0.0001);let nh=max(dot(n,h),0.0);let vh=max(dot(v,h),0.0);
 let a=max(r*r,0.002);let a2=a*a;let denominator=nh*nh*(a2-1.0)+1.0;
 let distribution=a2/(3.14159265*denominator*denominator);
 let gv=nl*sqrt(nv*nv*(1.0-a2)+a2);let gl=nv*sqrt(nl*nl*(1.0-a2)+a2);let visibility=0.5/max(gv+gl,0.00001);
 let f=fresnel(vh,f0);
 return ((1.0-f)*color/3.14159265+f*distribution*visibility)*nl;
}
fn lighting(s:Surface,color:vec3f,transmitted:vec3f,transmission:f32)->vec3f {
 let n=normalize(s.normal);let v=normalize(frame.eye.xyz-s.world);let r=reflect(-v,n);let rough=frame.optics.w;
 let f0=pow((frame.optics.y-1.0)/(frame.optics.y+1.0),2.0);
 // Thin-film iridescence (upstream: 0.12 strength, IOR 1.3, 180–420 nm).
 let phase=4.0f*3.14159265f*1.3f*mix(180.0f,420.0f,0.5f)*sqrt(max(0.0,1.0-(1.0-dot(n,v)*dot(n,v))/(1.3*1.3)));
 let film=0.5+0.5*cos(phase/vec3f(650.0,510.0,475.0));
 let specular=mix(vec3f(f0),vec3f(f0)*mix(vec3f(0.6),vec3f(1.4),film),0.12);
 let f=fresnel(abs(dot(n,v)),specular);
 let diffuse=color*mix(vec3f(0.40,0.39,0.36),vec3f(0.65,0.67,0.71),n.y*0.5+0.5)*0.6;
 var result=diffuse*(1.0-transmission)+transmitted*transmission;
 result+=environment(r,rough)*f*0.9;
 let opaque=color*(1.0-transmission);
 result+=brdf(n,v,normalize(vec3f(-3.0,7.0,5.0)-s.world),opaque,rough,specular)*vec3f(1.0,0.93,0.82)*1.8;
 result+=brdf(n,v,normalize(vec3f(5.0,4.0,-3.0)-s.world),opaque,rough,specular)*vec3f(0.58,0.76,1.0)*0.8;
 let coat=fresnel(abs(dot(n,v)),vec3f(0.04));result=result*(1.0-coat*0.15)+environment(r,0.075)*coat*0.15;
 let fog=smoothstep(18.0,45.0,length(view(s.world)));return mix(result,vec3f(0.815,0.791,0.723),fog);
}
fn floor_color(s:Surface)->vec3f {
 let distance=length(view(s.world));return mix(vec3f(0.70,0.67,0.61),vec3f(0.815,0.791,0.723),smoothstep(4.0,30.0,distance));
}
@fragment fn geometry(s:Surface)->@location(0) vec4f {
 coverage(s);let n=normalize(s.normal);return vec4f(dot(n,frame.right.xyz),dot(n,frame.up.xyz),-dot(n,frame.forward.xyz),-view(s.world).z);
}
@fragment fn peel(s:Surface)->@location(0) vec4f {
 coverage(s);let pixel=vec2i(s.clip.xy);let front=textureLoad(front_depth,pixel,0);
 if s.birth> -99999.0 && s.clip.z<=front+0.000003 {discard;}
 if s.birth< -99999.0 {return vec4f(floor_color(s),1.0);}
 return vec4f(lighting(s,pigment(s.birth),vec3f(0),0.0),1.0);
}
fn behind_at(uv:vec2f,roughness:f32,travel:f32,z:f32)->vec3f {
 let radius=roughness*roughness*(travel+0.35)*0.65;
 let footprint=radius*frame.viewport.w/max(abs(z)*2.0,0.001);
 var color=textureSampleLevel(behind_color,linear_sampler,uv,0.0).rgb;
 if roughness>0.001 {for(var i=0;i<8;i++) {let angle=f32(i)*0.785398;let offset=vec2f(cos(angle),sin(angle))*footprint;color+=textureSampleLevel(behind_color,linear_sampler,clamp(uv+offset,vec2f(0.001),vec2f(0.999)),0.0).rgb;}color/=9.0;}
 return color;
}
@fragment fn beauty(s:Surface)->@location(0) vec4f {
 coverage(s);if s.birth< -99999.0 {return vec4f(floor_color(s),1.0);}
 let uv=s.clip.xy/frame.viewport.xy;let front=textureLoad(front_depth,vec2i(s.clip.xy),0);
 let color=pigment(s.birth);let origin=view(s.world);let n=normalize(vec3f(dot(s.normal,frame.right.xyz),dot(s.normal,frame.up.xyz),-dot(s.normal,frame.forward.xyz)));
 var transmitted=vec3f(0);var transmission=0.0;
 if abs(s.clip.z-front)<0.00001 && frame.switches.x>0.0 {
  transmission=frame.switches.x;let incident=normalize(origin);let inside=refract(incident,n,1.0/frame.optics.y);
  let inside_distance=frame.optics.x/max(abs(dot(inside,n)),0.2);let exit=origin+inside*inside_distance;let end=exit+incident*8.0;
  let start_uv=project(exit);let end_uv=project(end);let delta=abs((end_uv-start_uv)*frame.viewport.xy);let steps=clamp(ceil(max(delta.x,delta.y)),16.0,384.0);
  let start_k=1.0/(-exit.z);let end_k=1.0/(-end.z);let start_q=exit*start_k;let end_q=end*end_k;
  var hit_uv=uv;var travel=inside_distance;var confidence=0.0;var previous_s=0.0;var previous_point=exit;var previous_front=true;
  for(var i=1;i<385;i++) {
   if f32(i)>steps {break;}let t=f32(i)/steps;let coord=mix(start_uv,end_uv,t);if any(coord<vec2f(0.001))||any(coord>vec2f(0.999)) {break;}
   let point=mix(start_q,end_q,t)/mix(start_k,end_k,t);let d=textureLoad(behind_depth,vec2i(coord*frame.viewport.xy),0);let surface=reconstruct(coord,d);
   let pixel_world=abs(surface.z)*2.0/(frame.viewport.y*frame.viewport.w);let tolerance=max(pixel_world*2.0,0.008);let error=surface.z-point.z;let step_length=length(point-previous_point);
   if d<0.99999 && error>=0.0 && previous_front && error<max(tolerance,step_length*1.5) && surface.z<origin.z-0.002 {
    var low=previous_s;var high=t;
    for(var j=0;j<8;j++) {let mid=(low+high)*0.5;let mid_uv=mix(start_uv,end_uv,mid);let mid_point=mix(start_q,end_q,mid)/mix(start_k,end_k,mid);let surface_z=reconstruct(mid_uv,textureLoad(behind_depth,vec2i(mid_uv*frame.viewport.xy),0)).z;if mid_point.z<surface_z {high=mid;}else {low=mid;}}
    let hit=(low+high)*0.5;let candidate=mix(start_uv,end_uv,hit);let hit_point=mix(start_q,end_q,hit)/mix(start_k,end_k,hit);let hit_surface=reconstruct(candidate,textureLoad(behind_depth,vec2i(candidate*frame.viewport.xy),0));let residual=abs(hit_point.z-hit_surface.z);
    if residual<tolerance {hit_uv=candidate;travel=length(hit_point-origin);confidence=1.0-smoothstep(tolerance*0.25,tolerance,residual);break;}
   }
   previous_front=point.z>surface.z;previous_s=t;previous_point=point;
  }
  let edge=min(min(hit_uv.x,hit_uv.y),min(1.0-hit_uv.x,1.0-hit_uv.y));confidence*=smoothstep(0.0,0.025,edge);
  let behind=mix(behind_at(uv,frame.optics.w,travel,origin.z),behind_at(hit_uv,frame.optics.w,travel,origin.z),confidence);
  let f0=pow((frame.optics.y-1.0)/(frame.optics.y+1.0),2.0);let f=f0+(1.0-f0)*pow(1.0-clamp(abs(dot(normalize(-origin),n)),0.0,1.0),5.0);
  transmitted=behind*mix(vec3f(1),color,0.95)*(1.0-f);
 }
 return vec4f(lighting(s,color,transmitted,transmission),1.0);
}
struct Quad {@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn quad(@builtin(vertex_index) i:u32)->Quad {var q:Quad;let p=vec2f(f32((i<<1u)&2u),f32(i&2u));q.position=vec4f(p*2.0-1.0,0,1);q.uv=vec2f(p.x,1.0-p.y);return q;}
fn aces(color:vec3f)->vec3f {
 // Three.js ACESFilmicToneMapping, including exposure / 0.6 and matrices.
 let input=mat3x3f(vec3f(0.59719,0.076,0.0284),vec3f(0.35458,0.90834,0.13383),vec3f(0.04823,0.01566,0.83777));
 let output=mat3x3f(vec3f(1.60475,-0.10208,-0.00327),vec3f(-0.53108,1.10813,-0.07276),vec3f(-0.07367,-0.00605,1.07602));
 let v=input*(color*0.9*exp2(frame.grade.x)/0.6);let a=v*(v+0.0245786)-0.000090537;let b=v*(0.983729*v+0.432951)+0.238081;return clamp(output*(a/b),vec3f(0),vec3f(1));
}
@fragment fn ambient(q:Quad)->@location(0) vec4f {
 let uv=q.uv;let pixel=vec2i(clamp(uv*frame.viewport.xy,vec2f(0),frame.viewport.xy-1.0));
 let depth=textureLoad(front_depth,pixel,0);if depth>=0.99999 || frame.switches.y==0.0 {return vec4f(1);}
 let p=reconstruct(uv,depth);let normal=normalize(textureLoad(normals,pixel,0).xyz);var occlusion=0.0;
 for(var direction=0;direction<8;direction++) {let angle=f32(direction)*0.785398;let axis=vec2f(cos(angle),sin(angle));
  for(var sample_index=1;sample_index<=2;sample_index++) {
   let offset=axis*(f32(sample_index)/2.0)*0.55*frame.viewport.w/max(-p.z,0.1)*0.5;
   let coord=clamp(uv+offset,vec2f(0.001),vec2f(0.999));let d=textureLoad(front_depth,vec2i(coord*frame.viewport.xy),0);
   let delta=reconstruct(coord,d)-p;let distance=length(delta);
   if distance>0.0001 && distance<0.6 {occlusion+=max(dot(normal,delta/distance)-0.05,0.0)*(1.0-distance/0.6);}
  }
 }
 return vec4f(vec3f(1.0-clamp(occlusion/16.0*3.0,0.0,1.0)),1);
}
fn graded(uv:vec2f)->vec3f {
 let pixel=vec2i(clamp(uv*frame.viewport.xy,vec2f(0),frame.viewport.xy-1.0));var color=textureSampleLevel(image,linear_sampler,uv,0.0).rgb;
 if frame.switches.y>0.0 {
  let center_depth=textureLoad(front_depth,pixel,0);let center_normal=textureLoad(normals,pixel,0).xyz;
  var total=0.0;var weight=0.0;
  for(var x=-1;x<=1;x++) {for(var y=-1;y<=1;y++) {
   let coord=clamp(pixel+vec2i(x,y),vec2i(0),vec2i(frame.viewport.xy)-1);
   let d=textureLoad(front_depth,coord,0);let n=textureLoad(normals,coord,0).xyz;
   let w=exp(-abs(d-center_depth)*2000.0)*pow(max(dot(n,center_normal),0.0),8.0)+0.0001;
   total+=textureLoad(ambient_texture,coord,0).r*w;weight+=w;
  }}
  color*=mix(1.0,total/weight,0.85);
 }
 var display=aces(color);display=select(display*12.92,1.055*pow(display,vec3f(1.0/2.4))-0.055,display>vec3f(0.0031308));
 let balanced=display*exp2(vec3f(0.22,0,-0.22)*frame.grade.w);let luminance=dot(balanced,vec3f(0.2126,0.7152,0.0722));return clamp((mix(vec3f(luminance),balanced,frame.grade.z)-0.5)*frame.grade.y+0.5,vec3f(0),vec3f(1));
}
@fragment fn post(q:Quad)->@location(0) vec4f { return vec4f(graded(q.uv),1); }
@fragment fn antialias(q:Quad)->@location(0) vec4f {
 let texel=1.0/frame.viewport.xy;let c=textureSampleLevel(graded_texture,linear_sampler,q.uv,0.0).rgb;let n=textureSampleLevel(graded_texture,linear_sampler,q.uv+vec2f(0,-texel.y),0.0).rgb;let s=textureSampleLevel(graded_texture,linear_sampler,q.uv+vec2f(0,texel.y),0.0).rgb;let e=textureSampleLevel(graded_texture,linear_sampler,q.uv+vec2f(texel.x,0),0.0).rgb;let w=textureSampleLevel(graded_texture,linear_sampler,q.uv+vec2f(-texel.x,0),0.0).rgb;
 let l=vec3f(0.299,0.587,0.114);let range=max(max(dot(n,l),dot(s,l)),max(dot(e,l),dot(w,l)))-min(min(dot(n,l),dot(s,l)),min(dot(e,l),dot(w,l)));
 let smoothed=select(c,mix(c,(n+s+e+w)*0.25,0.5),range>0.125);return vec4f(smoothed,1);
}
