"""Export approved dynamic context and actual camera, never static scene GT."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from prepare_context import load_model_input


def camera_from_render(metadata):
    width, height = map(int, metadata["resolution"])
    c = metadata["camera"]
    eye = np.asarray(c["location"],dtype=np.float64)
    forward = np.asarray(c["target"],dtype=np.float64)-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,np.array([0.,0.,1.]))
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Vertical look direction needs an explicit camera roll")
    right /= np.linalg.norm(right)
    up = np.cross(right,forward)
    rotation = np.stack((right,-up,forward))
    rt = np.column_stack((rotation,-rotation@eye))
    fov = float(c["effective_yfov_deg"])
    if not 0 < fov < 180:
        raise ValueError("Invalid vertical field of view")
    focal = height/(2*np.tan(np.deg2rad(fov)/2))
    k = np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    if "forward" in c and not np.allclose(forward,c["forward"],atol=2e-5):
        raise ValueError("Stored rendered camera forward and target disagree")
    return k,rt,(height,width)


def project(points,k,rt):
    camera = np.asarray(points)@rt[:,:3].T+rt[:,3]
    if np.any(camera[...,2] <= 0):
        raise ValueError("Observed object lies behind the camera")
    pixels = camera@k.T
    return pixels[...,:2]/pixels[...,2:],camera[...,2]


def size_from_actor(actor):
    size = actor["size_m"]
    shape = actor["shape"]
    if shape == "sphere":
        return np.repeat(float(size["radius"])*2,3)
    if shape in ("puck","cylinder"):
        return np.array([2*float(size["radius"]),2*float(size["radius"]),float(size["height"])])
    raise ValueError("This preflight only supports spheres and upright pucks/cylinders")


def observed_prompts(positions,size,k,rt,hw):
    signs = np.asarray(list(itertools.product((-1,1),repeat=3)))
    corners = positions[:,:,None]+size[None,:,None]*signs[None,None]*.5
    projected,_ = project(corners,k,rt)
    lo,hi = projected.min(2),projected.max(2)
    margin = np.maximum(2.,.05*(hi-lo))
    lo,hi = lo-margin,hi+margin
    height,width = hw
    lo = np.maximum(lo,[0.,0.])
    hi = np.minimum(hi,[width-1.,height-1.])
    if np.any(hi<=lo):
        raise ValueError("Empty observed object prompt")
    center,_ = project(positions,k,rt)
    if np.any(center<lo) or np.any(center>hi):
        raise ValueError("Observed object center outside image/prompt")
    return np.concatenate((lo,hi),-1),center


def extract_observed(sample,render,frame_times):
    metadata = json.loads((sample/"metadata.json").read_text())
    actors = metadata["actors"]
    dynamic = [name for name,actor in actors.items() if actor.get("dynamic") is True]
    if not 1 <= len(dynamic) <= 2:
        raise ValueError("Expected one or two context objects")
    with np.load(sample/"raw/states_xyzw.npz",allow_pickle=False) as states:
        names = states["object_names"].astype(str).tolist()
        ids = [names.index(name) for name in dynamic]
        positions = states["positions"][:8,ids].astype(np.float64)
        times = states["frame_times"][:8].astype(np.float64)
        if not np.allclose(times,frame_times,atol=1e-6,rtol=0):
            raise ValueError("Observation times disagree with rendered RGB8")
        for name,index in zip(dynamic,ids):
            if actors[name]["shape"] in ("puck","cylinder"):
                q = states["quats"][:8,index].astype(np.float64)
                z_dot = 1-2*(q[:,0]**2+q[:,1]**2)
                if np.any(np.abs(z_dot)<.999):
                    raise ValueError("Tilted cylinder needs an oriented context bound")
    size = np.stack([size_from_actor(actors[name]) for name in dynamic])
    if positions.shape != (8,len(dynamic),3) or not np.isfinite(positions).all() or not np.all(size>0):
        raise ValueError("Invalid dynamic context")
    k,rt,hw = camera_from_render(render)
    boxes,centers = observed_prompts(positions,size,k,rt,hw)
    diagnostics = render["camera"].get("object_projections_xy_depth",{})
    errors = []
    for j,name in enumerate(dynamic):
        if name not in diagnostics:
            raise ValueError(f"Missing rendered dynamic projection diagnostic: {name}")
        x,y,z = diagnostics[name]
        expected = np.array([x*hw[1]-.5,(1-y)*hw[0]-.5])
        _,depth = project(positions[0,j],k,rt)
        error = float(np.linalg.norm(centers[0,j]-expected))
        if error>.05 or abs(float(depth)-z)>1e-4:
            raise ValueError(f"Rendered camera reprojection mismatch for {name}: {error:.5f}px")
        errors.append(dict(object_name=name,pixel_error=error,depth_error_m=abs(float(depth)-z)))
    arrays = dict(positions_world=positions,size_m=size,camera_K=k,camera_world_to_view=rt,
                  mask_boxes_xyxy=boxes,center_pixels=centers,frame_times=times)
    return arrays,dict(dynamic_names=dynamic,dynamic_camera_check=errors,source_hw=list(hw),
                      static_scene_gt_exported=False,future_state_exported=False,
                      dynamic_material_exported=False,quaternions_exported=False,
                      position_size_and_camera_source="user-approved observed context / rendered camera",
                      bound_scope="sphere AABB or observed-upright cylinder; not an exact silhouette")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--sample",type=Path,required=True)
    parser.add_argument("--render-metadata",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rgb,times = load_model_input(args.input)
    render = json.loads(args.render_metadata.read_text())
    arrays,audit = extract_observed(args.sample,render,times)
    if list(rgb.shape[1:3]) != audit["source_hw"]:
        raise ValueError("Actual rendering resolution differs from RGB input")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"context_geometry.npz",**arrays)
    audit.update(input_json=str(args.input.resolve()),sample=str(args.sample.resolve()),
                 render_metadata=str(args.render_metadata.resolve()),
                 arrays={name:list(value.shape) for name,value in arrays.items()},
                 gt_static_geometry_used=False,gt_mask_used=False,training_ready=False)
    (args.output/"report.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()

