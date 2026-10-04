// Mesh checks shared by the node tests; faces are index arrays, V is [[x,y,z]]. Faces fan into
// triangles; two triangles that share no vertex cross when an edge of one passes strictly through
// the other, both ends off its plane by more than span·epsRel. Coplanar overlap is not a crossing.
export function triangulate(faces){const t=[];for(const f of faces)for(let i=1;i+1<f.length;i++)t.push([f[0],f[i],f[i+1]]);return t;}
export function signedVolume(V,faces){let s=0;for(const [a,b,c] of triangulate(faces)){const A=V[a],B=V[b],C=V[c];s+=A[0]*(B[1]*C[2]-B[2]*C[1])-A[1]*(B[0]*C[2]-B[2]*C[0])+A[2]*(B[0]*C[1]-B[1]*C[0]);}return s/6;}
export function topo(faces){const e=new Map(),d=new Map();for(const f of faces)for(let i=0;i<f.length;i++){const a=f[i],b=f[(i+1)%f.length];const k=a<b?a+','+b:b+','+a;e.set(k,(e.get(k)||0)+1);const dk=a+'>'+b;d.set(dk,(d.get(dk)||0)+1);}let bnd=0,nm=0,mis=0;for(const c of e.values()){if(c===1)bnd++;else if(c>2)nm++;}for(const [k,c] of d){if(c>1)mis++;}return {E:e.size,boundary:bnd,nonManifold:nm,misoriented:mis};}
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]],cr=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],dt=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
function segTri(p,q,A,B,C,eps){const n=cr(sub(B,A),sub(C,A));const dp=dt(n,sub(p,A)),dq=dt(n,sub(q,A));const L=Math.sqrt(dt(n,n));if(L===0)return false;if(dp*dq>=0)return false;if(Math.abs(dp)/L<eps||Math.abs(dq)/L<eps)return false;const t=dp/(dp-dq);const x=[p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1]),p[2]+t*(q[2]-p[2])];const c1=dt(cr(sub(B,A),sub(x,A)),n),c2=dt(cr(sub(C,B),sub(x,B)),n),c3=dt(cr(sub(A,C),sub(x,C)),n);return (c1>0&&c2>0&&c3>0);}
export function selfCrossings(V,faces,epsRel=1e-7){const T=triangulate(faces);let lo=[1/0,1/0,1/0],hi=[-1/0,-1/0,-1/0];for(const p of V)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}const span=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])||1;const eps=span*epsRel;
 const bb=T.map(t=>{const a=[1/0,1/0,1/0],b=[-1/0,-1/0,-1/0];for(const i of t)for(let k=0;k<3;k++){a[k]=Math.min(a[k],V[i][k]);b[k]=Math.max(b[k],V[i][k]);}return [a,b];});
 const G=Math.max(1,Math.round(Math.cbrt(T.length)));const cell=span/G;const grid=new Map();const key=(x,y,z)=>x+','+y+','+z;
 T.forEach((t,i)=>{const [a,b]=bb[i];const i0=a.map((v,k)=>Math.floor((v-lo[k])/cell)),i1=b.map((v,k)=>Math.floor((v-lo[k])/cell));for(let x=i0[0];x<=i1[0];x++)for(let y=i0[1];y<=i1[1];y++)for(let z=i0[2];z<=i1[2];z++){const kk=key(x,y,z);if(!grid.has(kk))grid.set(kk,[]);grid.get(kk).push(i);}});
 const seen=new Set();let n=0;const ex=[];
 for(const list of grid.values())for(let x=0;x<list.length;x++)for(let y=x+1;y<list.length;y++){const i=list[x],j=list[y];const k=i<j?i*1e7+j:j*1e7+i;if(seen.has(k))continue;seen.add(k);const a=T[i],b=T[j];if(a.some(v=>b.includes(v)))continue;const [ai,ax]=bb[i],[bi,bx]=bb[j];if(ai[0]>bx[0]||bi[0]>ax[0]||ai[1]>bx[1]||bi[1]>ax[1]||ai[2]>bx[2]||bi[2]>ax[2])continue;
  let hit=false;for(let e=0;e<3&&!hit;e++){hit=segTri(V[a[e]],V[a[(e+1)%3]],V[b[0]],V[b[1]],V[b[2]],eps)||segTri(V[b[e]],V[b[(e+1)%3]],V[a[0]],V[a[1]],V[a[2]],eps);}if(hit){n++;if(ex.length<3)ex.push([i,j]);}}
 return n;}
