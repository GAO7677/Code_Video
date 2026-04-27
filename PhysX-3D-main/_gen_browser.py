#!/usr/bin/env python3
# Generates urdf_browser.py
import os

PYTHON_HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
urdf_browser.py  -  PhysXNet URDF Folder Browser + 3-D Viewer

Usage:
    python urdf_browser.py \\
        --urdf_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/urdf \\
        --port 8021

Open http://127.0.0.1:8021 in your browser.
Remote: ssh -L 8021:127.0.0.1:8021 user@server
"""

import argparse, json, math, posixpath
import xml.etree.ElementTree as ET
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

def parse_vec3(s, default=(0.,0.,0.)):
    if not s: return list(default)
    v = [float(x) for x in s.strip().split()]
    return v if len(v)==3 else list(default)

def eye4(): return [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]

def mat4_mul(a,b):
    o=[[0.]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4): o[i][j]=sum(a[i][k]*b[k][j] for k in range(4))
    return o

def scale4(sx,sy,sz): return [[sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]]

def xform(xyz,rpy):
    roll,pitch,yaw=rpy
    cx,sx=math.cos(roll),math.sin(roll)
    cy,sy_=math.cos(pitch),math.sin(pitch)
    cz,sz=math.cos(yaw),math.sin(yaw)
    def m3(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    rot=m3(m3([[cz,-sz,0],[sz,cz,0],[0,0,1]],[[cy,0,sy_],[0,1,0],[-sy_,0,cy]]),[[1,0,0],[0,cx,-sx],[0,sx,cx]])
    T=eye4()
    for i in range(3):
        for j in range(3): T[i][j]=rot[i][j]
    T[0][3],T[1][3],T[2][3]=xyz
    return T

def col_major(m): return [m[r][c] for c in range(4) for r in range(4)]

def parse_urdf(p):
    root=ET.parse(p).getroot()
    links,lvis,joints,c2j=[],{},{},{}
    for lk in root.findall("link"):
        n=lk.attrib["name"]; links.append(n); lvis[n]=[]
        for vis in lk.findall("visual"):
            orig=vis.find("origin")
            xyz=parse_vec3(orig.attrib.get("xyz") if orig is not None else None)
            rpy=parse_vec3(orig.attrib.get("rpy") if orig is not None else None)
            geo=vis.find("geometry")
            if geo is None: continue
            mesh=geo.find("mesh")
            if mesh is None: continue
            mf=mesh.attrib.get("filename")
            if not mf: continue
            sc=parse_vec3(mesh.attrib.get("scale","1 1 1"),(1.,1.,1.))
            lvis[n].append((mf,sc,xform(xyz,rpy)))
    for jt in root.findall("joint"):
        pe=jt.find("parent"); ce=jt.find("child")
        if pe is None or ce is None: continue
        orig=jt.find("origin")
        xyz=parse_vec3(orig.attrib.get("xyz") if orig is not None else None)
        rpy=parse_vec3(orig.attrib.get("rpy") if orig is not None else None)
        e={"name":jt.attrib.get("name",""),"type":jt.attrib.get("type","fixed"),
           "parent":pe.attrib["link"],"child":ce.attrib["link"],"tf":xform(xyz,rpy)}
        joints[e["name"]]=e; c2j[e["child"]]=e
    return links,lvis,list(joints.values()),c2j

def world_tfs(links,c2j):
    w={}
    def solve(n):
        if n in w: return w[n]
        if n not in c2j: w[n]=eye4(); return w[n]
        j=c2j[n]; w[n]=mat4_mul(solve(j["parent"]),j["tf"]); return w[n]
    for l in links: solve(l)
    return w

PALETTE=["#e74c3c","#3498db","#2ecc71","#f1c40f","#9b59b6","#e67e22",
         "#1abc9c","#95a5a6","#c0392b","#2980b9","#27ae60","#f39c12",
         "#8e44ad","#d35400","#16a085","#7f8c8d"]

def build_manifest(urdf):
    links,lvis,joints,c2j=parse_urdf(urdf)
    w=world_tfs(links,c2j)
    try: rname=ET.parse(urdf).getroot().attrib.get("name","scene")
    except: rname="scene"
    items,paths=[],[]
    idx=0
    for li,ln in enumerate(links):
        for vi,(mf,sc,otf) in enumerate(lvis.get(ln,[])):
            mp=(urdf.parent/mf).resolve()
            if not mp.exists(): print(f"[WARN] missing: {mp}"); continue
            T=mat4_mul(w[ln],mat4_mul(otf,scale4(*sc)))
            paths.append(mp)
            items.append({"name":f"{ln}_v{vi}","url":f"/mesh/{idx}.obj",
                          "matrix":col_major(T),"color":PALETTE[li%len(PALETTE)]})
            idx+=1
    return {"robot_name":rname,"num_links":len(links),"num_joints":len(joints),
            "joints":[{"name":j["name"],"type":j["type"]} for j in joints],
            "items":items}, paths

class State:
    def __init__(self,urdf_dir):
        self.urdf_dir=Path(urdf_dir)
        self.all_urdfs=sorted(self.urdf_dir.glob("*.urdf"))
        self.names=[u.stem for u in self.all_urdfs]
        self._cache={}
    def get(self,stem):
        if stem in self._cache: return self._cache[stem]
        urdf=self.urdf_dir/(stem+".urdf")
        if not urdf.exists(): return None,None
        try:
            m,p=build_manifest(urdf); self._cache[stem]=(m,p); return m,p
        except Exception as e:
            print(f"[ERR] {stem}: {e}"); return None,None

'''

CSS = '''
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--accent:#58a6ff;
  --accent2:#f78166;--text:#c9d1d9;--muted:#8b949e;--hover:#1c2230;
  --r:7px;--mono:"JetBrains Mono",monospace;}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:"Inter",system-ui;overflow:hidden;}
body{display:flex;flex-direction:column;}
header{display:flex;align-items:center;gap:12px;padding:9px 18px;
  border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0;}
header h1{font-size:14px;font-weight:700;color:var(--accent);letter-spacing:.4px;}
.badge{font-size:10px;font-family:var(--mono);background:var(--border);
  color:var(--muted);padding:2px 7px;border-radius:20px;}
#search{margin-left:auto;background:var(--bg);border:1px solid var(--border);
  color:var(--text);padding:5px 11px;border-radius:var(--r);
  font-size:12px;width:200px;outline:none;font-family:var(--mono);}
#search:focus{border-color:var(--accent);}
.layout{display:flex;flex:1;overflow:hidden;}
.sidebar{width:200px;flex-shrink:0;border-right:1px solid var(--border);
  display:flex;flex-direction:column;background:var(--panel);}
.shdr{padding:7px 13px;font-size:9px;text-transform:uppercase;
  letter-spacing:1.3px;color:var(--muted);border-bottom:1px solid var(--border);flex-shrink:0;}
.file-list{overflow-y:auto;flex:1;}
.fi{padding:6px 13px;cursor:pointer;font-size:11px;font-family:var(--mono);
  transition:background .1s;display:flex;align-items:center;gap:7px;
  border-left:2px solid transparent;user-select:none;}
.fi:hover{background:var(--hover);}
.fi.active{background:rgba(88,166,255,.1);color:var(--accent);border-left-color:var(--accent);}
.dot{width:5px;height:5px;border-radius:50%;background:var(--border);flex-shrink:0;}
.fi.active .dot{background:var(--accent);}
.vpane{flex:1;display:flex;flex-direction:column;overflow:hidden;}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;
  padding:6px 12px;border-bottom:1px solid var(--border);
  background:var(--panel);flex-shrink:0;min-height:34px;}
.toolbar .ttl{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--accent2);}
.tag{font-size:10px;background:rgba(48,54,61,.9);
  padding:1px 7px;border-radius:10px;color:var(--muted);font-family:var(--mono);}
.tag b{color:var(--text);}
#lmsg{margin-left:auto;font-size:10px;color:var(--muted);font-family:var(--mono);}
.mid{display:flex;flex:1;overflow:hidden;}
.canvas-host{flex:1;overflow:hidden;position:relative;}
#c{width:100%;height:100%;display:block;}
.ipanel{width:185px;flex-shrink:0;border-left:1px solid var(--border);
  background:var(--panel);overflow-y:auto;padding:8px;}
.ipanel h3{font-size:9px;text-transform:uppercase;letter-spacing:1.3px;
  color:var(--muted);margin-bottom:5px;}
.sec{margin-bottom:12px;}
.litem{display:flex;align-items:center;gap:6px;font-size:10px;
  font-family:var(--mono);margin-bottom:2px;color:var(--muted);line-height:1.5;}
.sw{width:8px;height:8px;border-radius:2px;flex-shrink:0;}
.jitem{font-size:10px;font-family:var(--mono);margin-bottom:2px;color:var(--muted);
  display:flex;align-items:center;gap:3px;flex-wrap:wrap;}
.jt{font-size:9px;padding:0 4px;border-radius:3px;white-space:nowrap;}
.jt-revolute{background:rgba(88,166,255,.18);color:#58a6ff;}
.jt-prismatic{background:rgba(247,129,102,.18);color:#f78166;}
.jt-fixed{background:rgba(139,148,158,.12);color:#8b949e;}
.jt-floating{background:rgba(63,185,80,.18);color:#3fb950;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
'''

HTML_BODY = '''
<header>
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#58a6ff" stroke-width="2">
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
    <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
    <line x1="12" y1="22.08" x2="12" y2="12"/>
  </svg>
  <h1>PhysXNet URDF Browser</h1>
  <span class="badge" id="total-badge">...</span>
  <input id="search" type="text" placeholder="filter id..." autocomplete="off">
</header>
<div class="layout">
  <div class="sidebar">
    <div class="shdr">URDF Files</div>
    <div class="file-list" id="file-list"></div>
  </div>
  <div class="vpane">
    <div class="toolbar">
      <span class="ttl" id="obj-title">—</span>
      <span class="tag">links: <b id="t-links">—</b></span>
      <span class="tag">joints: <b id="t-joints">—</b></span>
      <span class="tag">meshes: <b id="t-meshes">—</b></span>
      <span id="lmsg">select a model from the list</span>
    </div>
    <div class="mid">
      <div class="canvas-host"><canvas id="c"></canvas></div>
      <div class="ipanel">
        <div class="sec"><h3>Parts / Links</h3><div id="legend"></div></div>
        <div class="sec"><h3>Joints</h3><div id="jlist"></div></div>
      </div>
    </div>
  </div>
</div>
'''

JS = '''
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/OBJLoader.js";
const canvas=document.getElementById("c");
const renderer=new THREE.WebGLRenderer({canvas,antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d1117);
const camera=new THREE.PerspectiveCamera(50,1,0.01,2000);
camera.position.set(2,1.5,2.5);
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,0.7));
const d1=new THREE.DirectionalLight(0xffffff,0.9);d1.position.set(3,4,5);scene.add(d1);
const d2=new THREE.DirectionalLight(0xffffff,0.4);d2.position.set(-3,2,-4);scene.add(d2);
scene.add(new THREE.AxesHelper(0.4));
scene.add(new THREE.GridHelper(4,20,0x444444,0x222222));
const host=document.querySelector(".canvas-host");
function resize(){const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe(host);resize();
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);}
animate();
let modelGroup=null;
function clearModel(){if(modelGroup){scene.remove(modelGroup);modelGroup=null;}}
function fitCamera(g){
  const box=new THREE.Box3().setFromObject(g);
  if(!isFinite(box.min.x))return;
  const sz=new THREE.Vector3(),ctr=new THREE.Vector3();
  box.getSize(sz);box.getCenter(ctr);
  const md=Math.max(sz.x,sz.y,sz.z)||1;
  const dist=Math.abs(md/2/Math.tan(camera.fov*Math.PI/360))*2.2;
  camera.position.set(ctr.x+dist*0.6,ctr.y+dist*0.4,ctr.z+dist);
  camera.near=md/2000;camera.far=md*2000;camera.updateProjectionMatrix();
  controls.target.copy(ctr);controls.update();
}
const objLoader=new OBJLoader();
async function loadModel(stem){
  document.getElementById("lmsg").textContent="loading...";
  clearModel();
  const manifest=await fetch("/manifest.json?id="+stem).then(r=>r.json());
  if(manifest.error){document.getElementById("lmsg").textContent="error: "+manifest.error;return;}
  document.getElementById("obj-title").textContent=stem;
  document.getElementById("t-links").textContent=manifest.num_links;
  document.getElementById("t-joints").textContent=manifest.num_joints;
  document.getElementById("t-meshes").textContent=manifest.items.length;
  const leg=document.getElementById("legend");leg.innerHTML="";
  const seen=new Set();
  manifest.items.forEach(it=>{
    const base=it.name.replace(/_v\\d+$/,"");
    if(seen.has(base))return;seen.add(base);
    const d=document.createElement("div");d.className="litem";
    d.innerHTML="<div class=\'sw\' style=\'background:"+it.color+"\'/><span>"+base+"</span>";
    leg.appendChild(d);
  });
  const jl=document.getElementById("jlist");jl.innerHTML="";
  manifest.joints.forEach(j=>{
    const d=document.createElement("div");d.className="jitem";
    d.innerHTML="<span>"+j.name+"</span><span class=\'jt jt-"+j.type+"\'/>"+j.type+"</span>";
    jl.appendChild(d);
  });
  modelGroup=new THREE.Group();scene.add(modelGroup);
  let done=0,total=manifest.items.length;
  document.getElementById("lmsg").textContent="0/"+total;
  for(const item of manifest.items){
    await new Promise(resolve=>{
      objLoader.load(item.url,obj=>{
        const mat=new THREE.Matrix4().fromArray(item.matrix);
        obj.traverse(ch=>{if(ch.isMesh)ch.material=new THREE.MeshStandardMaterial({color:item.color,metalness:0.05,roughness:0.75});});
        obj.matrixAutoUpdate=false;obj.applyMatrix4(mat);modelGroup.add(obj);
        done++;document.getElementById("lmsg").textContent=done+"/"+total;resolve();
      },undefined,()=>{done++;resolve();});
    });
  }
  fitCamera(modelGroup);
  document.getElementById("lmsg").textContent="done";
}
let allNames=[],current=null;
async function initSidebar(){
  const data=await fetch("/filelist.json").then(r=>r.json());
  allNames=data.names;
  document.getElementById("total-badge").textContent=allNames.length+" files";
  renderList(allNames);
}
function renderList(names){
  const fl=document.getElementById("file-list");fl.innerHTML="";
  names.forEach(name=>{
    const d=document.createElement("div");d.className="fi"+(name===current?" active":"");
    d.innerHTML="<div class=\'dot\'></div>"+name;
    d.onclick=()=>selectModel(name);fl.appendChild(d);
  });
}
function selectModel(name){
  current=name;
  document.querySelectorAll(".fi").forEach(el=>el.classList.toggle("active",el.textContent.trim()===name));
  loadModel(name);
}
document.getElementById("search").addEventListener("input",e=>{
  const q=e.target.value.trim().toLowerCase();
  renderList(q?allNames.filter(n=>n.includes(q)):allNames);
});
initSidebar();
'''

PYTHON_SERVER = '''
def make_page():
    return (
        "<!doctype html>\\n<html lang=\'en\'>\\n<head>\\n"
        "<meta charset=\'utf-8\'>\\n"
        "<title>PhysXNet URDF Browser</title>\\n"
        "<link href=\'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap\' rel=\'stylesheet\'>\\n"
        "<style>" + CSS + "</style>\\n"
        "</head>\\n<body>\\n" + HTML_BODY +
        "<script type=\'module\'>" + JS + "</script>\\n"
        "</body>\\n</html>\\n"
    )

PAGE = make_page()

class Handler(BaseHTTPRequestHandler):
    state = None
    def _send(self, data, ct):
        if isinstance(data, str): data = data.encode()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._send(PAGE, "text/html; charset=utf-8")
        elif path == "/filelist.json":
            self._send(json.dumps({"names": self.state.names}), "application/json")
        elif path == "/manifest.json":
            stem = (qs.get("id") or [""])[0]
            if not stem:
                self._send(json.dumps({"error": "no id"}), "application/json")
                return
            manifest, _ = self.state.get(stem)
            if manifest is None:
                self._send(json.dumps({"error": f"not found: {stem}"}), "application/json")
                return
            self._send(json.dumps(manifest), "application/json")
        elif path.startswith("/mesh/") and path.endswith(".obj"):
            stem = (qs.get("id") or [""])[0]
            if not stem:
                self.send_error(400, "missing id"); return
            _, paths = self.state.get(stem)
            if paths is None:
                self.send_error(404, "model not found"); return
            try:
                idx = int(posixpath.basename(path)[:-4])
            except ValueError:
                self.send_error(400, "bad index"); return
            if idx < 0 or idx >= len(paths):
                self.send_error(404, "mesh index out of range"); return
            self._send(paths[idx].read_bytes(), "text/plain")
        else:
            self.send_error(404)
    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    state = State(args.urdf_dir)
    print(f"[INFO] Found {len(state.names)} URDF files in {args.urdf_dir}")

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[INFO] Serving at {url}")
    print(f"[INFO] Remote tunnel: ssh -L {args.port}:127.0.0.1:{args.port} user@server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n[INFO] Stopped.")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
'''

with open("/home/gaoya/Code_Video/PhysX-3D-main/urdf_browser.py", "w") as f:
    f.write(PYTHON_HEADER)
    f.write(PYTHON_SERVER)

print("urdf_browser.py written successfully")
