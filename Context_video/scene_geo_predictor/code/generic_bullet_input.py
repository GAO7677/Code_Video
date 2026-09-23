"""Family-free sphere/finite-mesh Bullet adapter; no dataset imports or alignment."""
import numpy as np

def run(state,primitives,physics,gravity,*,advance=True,geometry_builder=None,diagnostic_alignment=None,log_api_contacts=False):
    import pybullet as p
    if state.get('status')!='ESTIMATED':return {'status':'FAIL','reason':'unreliable_dynamic_state','step_calls':0}
    if advance and gravity.get('status')!='ESTIMATED':return {'status':'UNKNOWN','reason':'gravity_not_observed','step_calls':0}
    if state['shape']!='sphere':return {'status':'UNSUPPORTED','reason':'dynamic_shape','step_calls':0}
    client=p.connect(p.DIRECT)
    try:
        p.setPhysicsEngineParameter(fixedTimeStep=physics['fixedTimeStep'],numSubSteps=physics['numSubSteps'],
            numSolverIterations=120,contactERP=.2,erp=.2,physicsClientId=client)
        if advance:p.setGravity(*gravity['vector'],physicsClientId=client)
        if geometry_builder is not None:geometry_builder(p,client)
        for primitive in primitives:
            if primitive['type']!='triangle_mesh':raise ValueError('unsupported primitive')
            shape=p.createCollisionShape(p.GEOM_MESH,vertices=primitive['vertices'],indices=np.array(primitive['faces']).ravel().tolist(),
                                        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,physicsClientId=client)
            body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=primitive['position'],
                                   baseOrientation=primitive['orientation'],physicsClientId=client)
            p.changeDynamics(body,-1,lateralFriction=physics['friction'],restitution=physics['restitution'],physicsClientId=client)
        shape=p.createCollisionShape(p.GEOM_SPHERE,radius=state['radius'],physicsClientId=client)
        ball=p.createMultiBody(baseMass=physics['mass'],baseCollisionShapeIndex=shape,basePosition=state['p7'],physicsClientId=client)
        p.changeDynamics(ball,-1,lateralFriction=physics['friction'],restitution=physics['restitution'],
                         linearDamping=physics['linear_damping'],angularDamping=physics['angular_damping'],physicsClientId=client)
        p.resetBaseVelocity(ball,linearVelocity=state['v7'],angularVelocity=state['omega'],physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)
        initial=p.getContactPoints(bodyA=ball,physicsClientId=client)
        penetration=max([max(0.,-c[8]) for c in initial],default=0.)
        result={'status':'INITIAL_CHECK_ONLY','initial_penetration_m':penetration,'initial_overlap':penetration>.001,'step_calls':0,
                'initial_contact_count':len(initial),'position_modified':False}
        if not advance:return result
        if diagnostic_alignment is not None:
            from mesh_contact_alignment import admit
            audit=admit(p,client,ball,**diagnostic_alignment)
            result['initialization']=audit
            result['position_modified']=audit['pose_resets']>0
            if audit['status']=='BLOCKED_INITIAL_CONTACT':return dict(result,status='FAIL',reason=audit['reason'])
            penetration=audit['after_penetration_m']
        if penetration>.001:return dict(result,status='FAIL',reason='initial_overlap')
        positions=[];velocities=[];contacts=[];api_contacts=[]
        def detail():
            return [{'body':c[2],'distance':c[8],'normal':list(c[7]),'position_on_surface':list(c[6])}
                    for c in p.getContactPoints(bodyA=ball,physicsClientId=client)]
        if log_api_contacts:result['initial_contacts_detailed']=detail()
        for t in range(physics['outputs']):
            for _ in range(physics['calls_per_output']):
                p.stepSimulation(physicsClientId=client);result['step_calls']+=1
                if log_api_contacts:api_contacts.append(detail())
            positions.append(p.getBasePositionAndOrientation(ball,physicsClientId=client)[0]);velocities.append(p.getBaseVelocity(ball,physicsClientId=client)[0])
            contacts.append([{'body':c[2],'distance':c[8]} for c in p.getContactPoints(bodyA=ball,physicsClientId=client)])
        if log_api_contacts:result.update(api_contacts=api_contacts,contact_log_hz=1/physics['fixedTimeStep'],hidden_internal_substeps_observed=False)
        return dict(result,status='EXECUTED',positions=positions,velocities=velocities,contacts=contacts)
    finally:p.disconnect(physicsClientId=client)
