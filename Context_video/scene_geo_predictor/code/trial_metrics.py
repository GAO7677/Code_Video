"""Trajectory diagnostics; static truth is evaluation-only, never model input."""
from __future__ import annotations

import numpy as np


def rotation_xyzw(q):
    q = np.asarray(q, dtype=float)
    q = q/np.linalg.norm(q)
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def sphere_distances(trajectory, radius, boxes):
    """Signed sphere/solid separation for the ground and individual OBBs."""
    values = [trajectory[:,2]-radius]
    for center, half_size, quaternion in boxes:
        local = (trajectory-center)@rotation_xyzw(quaternion)
        delta = np.abs(local)-half_size
        sdf = np.linalg.norm(np.maximum(delta, 0), axis=-1)+np.minimum(delta.max(-1), 0)
        values.append(sdf-radius)
    return np.stack(values, axis=1)


def first_contact_onset(distance, epsilon=.005):
    contact = distance <= epsilon
    transition = contact[1:] & ~contact[:-1]
    future = np.flatnonzero(transition[7:].any(1))
    return int(future[0]+8) if len(future) else None


def sustained(mask, count=3):
    return bool(np.any(np.convolve(np.asarray(mask, dtype=int), np.ones(count, dtype=int), mode='valid') == count))


def behaviour(prediction, history, truth, sphere_radius=None, boxes=()):
    pred = np.concatenate((history, prediction), axis=0)
    gt = np.concatenate((history, truth), axis=0)
    pv, tv = np.diff(pred, axis=0)*30, np.diff(gt, axis=0)*30
    direction = history[-1,:2]-history[-2,:2]
    magnitude = np.linalg.norm(direction)
    result = dict(ADE_m=float(np.linalg.norm(prediction-truth, axis=-1).mean()),
                  FDE_m=float(np.linalg.norm(prediction[-1]-truth[-1])),
                  velocity_MAE_mps=float(np.abs(pv[7:]-tv[7:]).mean()))
    for name, trajectory, velocity in [('pred',pred,pv), ('gt',gt,tv)]:
        future_v = velocity[7:]
        result[name+'_stop_proxy'] = sustained(np.linalg.norm(future_v, axis=-1) < .1)
        result[name+'_descent_15cm_proxy'] = bool((trajectory[8:,2] < trajectory[7,2]-.15).any())
        result[name+'_reversal_proxy'] = sustained(future_v[:,:2]@(direction/max(magnitude,1e-12)) < -.1) if magnitude > 1e-6 else None
    if sphere_radius is None:
        result['surface_metrics'] = 'unsupported non-sphere: no orientation prediction'
        return result
    pd = sphere_distances(pred, sphere_radius, boxes)
    td = sphere_distances(gt, sphere_radius, boxes)
    pe, te = first_contact_onset(pd), first_contact_onset(td)
    result.update(surface_metrics='30Hz geometry-derived contact onset, not impulse or substep collision ground truth',
        pred_contact_onset_frame=pe, gt_contact_onset_frame=te,
        contact_onset_classification_correct=(pe is None) == (te is None),
        contact_onset_time_error_s=abs(pe-te)/30 if pe is not None and te is not None else None,
        pred_penetration_frame_rate=float((pd[8:] < -.005).any(1).mean()),
        gt_penetration_frame_rate=float((td[8:] < -.005).any(1).mean()),
        pred_max_penetration_m=float(max(0, -pd[8:].min())),
        gt_max_penetration_m=float(max(0, -td[8:].min())))
    result['post_contact_velocity_angle_deg'] = None
    if te is not None and te < 48:
        a, b = pv[te:min(te+3,48)].mean(0), tv[te:min(te+3,48)].mean(0)
        if np.linalg.norm(a) >= .05 and np.linalg.norm(b) >= .05:
            cosine = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            result['post_contact_velocity_angle_deg'] = float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    return result
