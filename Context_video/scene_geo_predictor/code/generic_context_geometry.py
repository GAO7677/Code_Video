"""Anonymous RGB-only observation mechanisms. No dataset or GT dependencies."""
import cv2
import numpy as np
from scipy.optimize import least_squares

def motion_prompt(rgb):
    gray=np.stack([cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb]).astype(np.float32)
    # Translation registration is a diagnostic only; camera motion is not silently ignored.
    shifts=[cv2.phaseCorrelate(gray[0],frame)[0] for frame in gray[1:]]
    if np.max(np.linalg.norm(shifts,axis=1))>3:
        raise ValueError('UNKNOWN_camera_motion_exceeds_translation_gate')
    change=np.abs(gray[-1]-np.median(gray[:-1],axis=0))
    med=np.median(change);mad=np.median(np.abs(change-med))
    threshold=max(12.,float(med+6*1.4826*mad))
    binary=(change>threshold).astype(np.uint8)
    binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary)
    candidates=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<12 or area>change.size*.1:continue
        score=float(change[labels==i].sum())
        candidates.append({'box':[max(0,x-w*.2),max(0,y-h*.2),min(rgb.shape[2]-1,x+w*1.2),min(rgb.shape[1]-1,y+h*1.2)],'score':score,'area':area})
    candidates.sort(key=lambda r:r['score'],reverse=True)
    audit={'threshold':threshold,'translation_shifts':shifts,'candidates':candidates,'discovery_uses_circularity':False}
    if not candidates:raise ValueError('FAIL_no_motion_candidate')
    if len(candidates)>1 and candidates[1]['score']>.8*candidates[0]['score']:
        raise ValueError('UNKNOWN_multiple_motion_candidates')
    return np.array(candidates[0]['box'],np.float32),audit

def sphere_fit(depth,k,e,masks,scale,times):
    centers=[];surfaces=[];radii=[];support=[]
    for t,mask in enumerate(masks):
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if len(contours)!=1:raise ValueError('UNKNOWN_fragmented_target_mask')
        contour=contours[0];area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
        circularity=4*np.pi*area/max(perimeter**2,1)
        if circularity<.65:raise ValueError('UNSUPPORTED_non_sphere_silhouette')
        y,x=np.where(cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0)
        if len(x)<20:raise ValueError('FAIL_insufficient_visible_surface')
        pixels=np.column_stack([x,y,np.ones(len(x))]);cam=(pixels@np.linalg.inv(k[t]).T)*depth[t,y,x,None]*scale
        xyz=(cam-e[t,:,3]*scale)@e[t,:,:3]
        # Algebraic sphere initializes nonlinear joint fit; no radius prior.
        A=np.column_stack([2*xyz,np.ones(len(xyz))]);b=(xyz**2).sum(axis=1)
        sol=np.linalg.lstsq(A,b,rcond=None)[0];r2=sol[3]+sol[:3]@sol[:3]
        if r2<=0:raise ValueError('FAIL_nonpositive_sphere_initialization')
        centers.append(sol[:3]);radii.append(np.sqrt(r2));surfaces.append(xyz)
        support.append({'pixels':len(x),'circularity':float(circularity),'individual_radius_m':float(np.sqrt(r2))})
    radius=float(np.median(radii));initial=np.r_[np.array(centers).ravel(),radius]
    def residual(v):
        return np.concatenate([np.linalg.norm(x-v[t*3:t*3+3],axis=1)-v[-1] for t,x in enumerate(surfaces)])
    lower=np.full(25,-np.inf);lower[-1]=1e-6
    fit=least_squares(residual,initial,bounds=(lower,np.full(25,np.inf)),loss='soft_l1',f_scale=max(radius*.02,1e-5),max_nfev=200)
    c=fit.x[:-1].reshape(8,3);r=float(fit.x[-1]);errors=residual(fit.x)
    singular=np.linalg.svd(fit.jac,compute_uv=False);condition=float(singular[0]/max(singular[-1],1e-15))
    radius_cv=float(np.std(radii)/max(np.mean(radii),1e-10));relative_rms=float(np.sqrt(np.mean(errors**2))/r)
    # Shared radius does not imply same height or support.
    tt=np.array(times)-times[-1];design=np.column_stack([np.ones(8),tt]);beta=np.linalg.lstsq(design,c,rcond=None)[0]
    motion=least_squares(lambda x:(design@x.reshape(2,3)-c).ravel(),beta.ravel(),loss='soft_l1',f_scale=max(r*.05,1e-5))
    beta=motion.x.reshape(2,3)
    reasons=[]
    if not fit.success:reasons.append('sphere_optimizer_failed')
    if relative_rms>.15:reasons.append('surface_residual_exceeds_15pct_radius')
    if radius_cv>.25:reasons.append('per_frame_radius_cv_exceeds_25pct')
    if condition>1e5:reasons.append('sphere_fit_ill_conditioned')
    if r>3*np.median(radii):reasons.append('joint_radius_diverged')
    return {'status':'FAIL' if reasons else 'ESTIMATED','reasons':reasons,'center_sequence':c.tolist(),'p7':beta[0].tolist(),'v7':beta[1].tolist(),
            'shape':'sphere','radius':r,'orientation':{'status':'UNKNOWN','collision_quaternion':[0,0,0,1],'reason':'sphere collision rotational symmetry'},
            'omega':[0,0,0],'omega_source':'explicit_zero_baseline','relative_rms':relative_rms,'radius_cv':radius_cv,'jacobian_condition':condition,'frames':support}

def finite_mesh(depth,k,e,masks,scale):
    # Reference-view observed surface only; image-adjacency triangulation cannot bridge masked holes.
    h,w=depth.shape[1:];union=cv2.dilate(np.any(masks,axis=0).astype(np.uint8),np.ones((9,9),np.uint8))>0
    yy,xx=np.mgrid[0:h:3,0:w:3];z=depth[0,yy,xx];valid=(z>0)&np.isfinite(z)&~union[yy,xx]
    pixels=np.stack([xx,yy,np.ones_like(xx)],axis=-1)
    cam=(pixels@np.linalg.inv(k[0]).T)*z[...,None]*scale
    xyz=(cam-e[0,:,3]*scale)@e[0,:,:3];vertices=xyz.reshape(-1,3);gh,gw=z.shape
    faces=[]
    for y in range(gh-1):
        for x in range(gw-1):
            for coords in [[(y,x),(y+1,x),(y,x+1)],[(y+1,x),(y+1,x+1),(y,x+1)]]:
                if not all(valid[a,b] for a,b in coords):continue
                zs=np.array([z[a,b] for a,b in coords])
                if np.ptp(zs)>.03*np.median(zs):continue
                ids=[a*gw+b for a,b in coords];faces.append(ids)
    if len(faces)<100:raise ValueError('FAIL_insufficient_finite_surface_triangles')
    triangles=vertices[np.array(faces)];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);length=np.linalg.norm(normals,axis=1)
    normals=normals/np.maximum(length[:,None],1e-12)
    # Geometry supplies axes, not a gravity label or sign. Do not manufacture floor semantics.
    sample=normals[::max(1,len(normals)//2000)];_,_,axes=np.linalg.svd(sample,full_matrices=False)
    gravity={'status':'UNKNOWN','vector':None,'reason':'observed surface normals do not uniquely identify gravity axis and sign; no floor/support assumption permitted',
             'unsigned_normal_axes':axes.tolist(),'axis_is_gravity':False}
    return {'type':'triangle_mesh','position':[0,0,0],'orientation':[0,0,0,1],'size':None,
            'confidence':{'status':'UNVALIDATED','reason':'observed depth mesh; no invisible completion'},
            'vertices':vertices.tolist(),'faces':faces,'thickness':{'status':'UNKNOWN','value':None},
            'source':'reference RGB0 VGGT depth; dynamic-mask union removed; stride3; relative depth jump<=3pct'},gravity
