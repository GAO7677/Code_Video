"""Pre-step admission and explicit, bounded support alignment for sphere pilots."""
from __future__ import annotations


class InitialContactError(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__(f"Initial contact rejected: {audit['reason']}")


def admit_initial_contact(p, client, ball, body_names, *, radius,
                          policy='strict', confirmed_support_names=()):
    if policy not in {'strict', 'support_aligned'}:
        raise ValueError(f'Unknown initialization policy: {policy}')
    tolerance = .001
    slop = .00001
    position, quaternion = p.getBasePositionAndOrientation(ball, physicsClientId=client)
    linear, angular = p.getBaseVelocity(ball, physicsClientId=client)

    def contacts():
        p.performCollisionDetection(physicsClientId=client)
        return list(p.getContactPoints(bodyA=ball, physicsClientId=client))

    def penetration(rows):
        return max([max(0., -float(row[8])) for row in rows], default=0.)

    rows = contacts()
    audit = dict(policy=policy, status='UNCHANGED', before_position_m=list(position),
                 after_position_m=list(position), before_penetration_m=penetration(rows),
                 after_penetration_m=penetration(rows), displacement_m=[0.,0.,0.],
                 max_alignment_m=radius*.5, tolerance_m=tolerance,
                 confirmed_support_names=sorted(confirmed_support_names),
                 support_bodies=[], api_step_calls_before_decision=0,
                 initialization_pose_resets=0, initialization_velocity_restores=0)

    def reject(reason):
        audit.update(status='BLOCKED_INITIAL_CONTACT', reason=reason)
        raise InitialContactError(audit)

    if audit['before_penetration_m'] <= tolerance:
        if confirmed_support_names:
            support = [row for row in rows if body_names.get(row[2]) in confirmed_support_names
                       and abs(float(row[8])) <= tolerance and float(row[7][2]) >= .99]
            if not support:
                reject('missing_confirmed_initial_support')
            for row in support:
                low, high = p.getAABB(row[2], physicsClientId=client)
                if not all(low[i] <= position[i] <= high[i] for i in (0, 1)):
                    reject('center_outside_support_footprint')
            audit['support_bodies'] = sorted({body_names[row[2]] for row in support})
        return audit
    if policy == 'strict':
        reject('initial_overlap_exceeds_tolerance')
    penetrating = [row for row in rows if float(row[8]) < -tolerance]
    if not penetrating or any(body_names.get(row[2]) not in confirmed_support_names for row in penetrating):
        reject('penetration_is_not_confirmed_support')
    if any(float(row[7][2]) < .99 for row in penetrating):
        reject('support_contact_is_not_upward_horizontal')
    if abs(float(linear[2])) > .01:
        reject('initial_state_is_not_stationary_in_support_normal')
    # Only horizontal box/plane tops are admitted by the upward-normal check.
    # No horizontal motion, velocity correction, pre-roll or solver retuning.
    tops = [float(p.getAABB(row[2], physicsClientId=client)[1][2]) for row in penetrating]
    delta = max(tops) + radius - slop - position[2]
    if delta <= 0:
        reject('alignment_is_not_upward')
    if delta > radius*.5:
        reject('correction_exceeds_half_radius')
    adjusted = [position[0], position[1], position[2]+delta]
    p.resetBasePositionAndOrientation(ball, adjusted, quaternion, physicsClientId=client)
    p.resetBaseVelocity(ball, linearVelocity=linear, angularVelocity=angular, physicsClientId=client)
    audit.update(initialization_pose_resets=1, initialization_velocity_restores=1,
                 after_position_m=adjusted, displacement_m=[0.,0.,delta],
                 support_bodies=sorted({body_names[row[2]] for row in penetrating}))
    after = contacts()
    audit['after_penetration_m'] = penetration(after)
    if audit['after_penetration_m'] > tolerance:
        reject('alignment_still_intersects_geometry')
    if not any(body_names.get(row[2]) in confirmed_support_names and abs(float(row[8])) <= tolerance and float(row[7][2]) >= .99 for row in after):
        reject('alignment_lost_support_contact')
    audit['status'] = 'CONTACT_ALIGNED'
    return audit

