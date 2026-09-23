"""Explicit C-only diagnostic alignment; no pre-roll, suction or shape changes."""
import numpy as np


def admit(p,client,ball,*,up,confirmed_support,aligned=False):
    up=np.asarray(up,dtype=float);up/=np.linalg.norm(up)
    position,quat=p.getBasePositionAndOrientation(ball,physicsClientId=client)
    linear,angular=p.getBaseVelocity(ball,physicsClientId=client)
    def contacts():
        p.performCollisionDetection(physicsClientId=client)
        return p.getContactPoints(bodyA=ball,physicsClientId=client)
    rows=contacts()
    penetration=lambda cs:max([max(0.,-c[8]) for c in cs],default=0.)
    audit={'policy':'support_aligned_mesh' if aligned else 'strict','status':'UNCHANGED',
        'before_position_m':list(position),'after_position_m':list(position),'displacement_m':[0.,0.,0.],
        'before_penetration_m':penetration(rows),'after_penetration_m':penetration(rows),
        'tolerance_m':.001,'max_alignment_m':.055,'confirmed_support_from_GT_A':bool(confirmed_support),
        'api_step_calls_before_decision':0,'pose_resets':0}
    def reject(reason):
        audit.update(status='BLOCKED_INITIAL_CONTACT',reason=reason)
        return audit
    if penetration(rows)<=.001:
        if aligned and confirmed_support and not any(abs(c[8])<=.001 and np.dot(c[7],up)>=.99 for c in rows):
            return reject('missing_confirmed_initial_support_no_suction')
        return audit
    if not aligned:return reject('initial_overlap_exceeds_tolerance')
    if not confirmed_support:return reject('support_not_confirmed_by_GT_A')
    if abs(np.dot(linear,up))>.01:return reject('nonstationary_support_normal_velocity')
    total=0.
    for _ in range(8):
        bad=[c for c in rows if c[8]<-.001]
        if any(np.dot(c[7],up)<.99 for c in bad):return reject('side_contact_not_horizontal_support')
        if not bad:break
        delta=max((-c[8]+.00001)/np.dot(c[7],up) for c in bad)
        if total+delta>.055:return reject('correction_exceeds_55mm')
        total+=delta;adjusted=np.asarray(position)+total*up
        p.resetBasePositionAndOrientation(ball,adjusted.tolist(),quat,physicsClientId=client)
        p.resetBaseVelocity(ball,linearVelocity=linear,angularVelocity=angular,physicsClientId=client)
        rows=contacts();audit.update(after_position_m=adjusted.tolist(),displacement_m=(total*up).tolist(),
            after_penetration_m=penetration(rows),pose_resets=audit['pose_resets']+1)
    if penetration(rows)>.001:return reject('alignment_still_intersects')
    if not any(abs(c[8])<=.001 and np.dot(c[7],up)>=.99 for c in rows):return reject('alignment_lost_support')
    audit['status']='CONTACT_ALIGNED'
    return audit
