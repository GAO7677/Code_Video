"""Observation-only recovery mechanisms; no dataset names or GT imports."""
import cv2
import numpy as np
from scipy.optimize import least_squares


def fixed_camera(rgb, masks, k, e):
    """Admit a fixed-camera prior from static feature tracks; no guessed calibration."""
    gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in rgb]
    exclusion=cv2.dilate(np.any(masks,axis=0).astype(np.uint8),np.ones((15,15),np.uint8))
    points=cv2.goodFeaturesToTrack(gray[0],200,.02,8,mask=(1-exclusion)*255)
    if points is None or len(points)<30:raise ValueError('insufficient_static_camera_features')
    shifts=[]
    for im in gray[1:]:
        q,ok,err=cv2.calcOpticalFlowPyrLK(gray[0],im,points,None)
        valid=(ok[:,0]>0)&(err[:,0]<15)
        if valid.sum()<30:raise ValueError('insufficient_static_camera_tracks')
        shifts.append(float(np.percentile(np.linalg.norm(q[valid,0]-points[valid,0],axis=1),90)))
    if max(shifts)>1.5:raise ValueError('fixed_camera_prior_not_supported')
    shared=np.median(k,axis=0);shared[2]=[0,0,1]
    return np.repeat(shared[None],8,axis=0),np.repeat(e[:1],8,axis=0),{'static_track_p90_px':shifts,'intrinsic_source':'clip median VGGT','extrinsic_source':'reference VGGT pose','absolute_calibration':'UNVALIDATED'}


def regularize_planes(mesh):
    """Fit bounded observed planar patches; keep faces/holes and cap movement at 1cm."""
    vertices=np.array(mesh['vertices']);faces=np.array(mesh['faces']);used=np.unique(faces)
    remaining=used.copy();rng=np.random.default_rng(42);patches=[]
    # No normals or coordinates specific to floor, family, or target state.
    for _ in range(8):
        if len(remaining)<200:break
        sample=remaining[rng.choice(len(remaining),min(4000,len(remaining)),replace=False)]
        pts=vertices[sample];best=None;count=0
        for _ in range(100):
            a,b,c=pts[rng.choice(len(pts),3,replace=False)];n=np.cross(b-a,c-a);length=np.linalg.norm(n)
            if length<1e-8:continue
            n/=length;inside=abs((pts-a)@n)<.01
            if inside.sum()>count:best=(n,float(-n@a));count=int(inside.sum())
        if count<100:break
        n,offset=best;selected=remaining[abs(vertices[remaining]@n+offset)<.01]
        center=vertices[selected].mean(axis=0);_,_,v=np.linalg.svd(vertices[selected]-center,full_matrices=False);n=v[-1];offset=-float(n@center)
        signed=vertices[selected]@n+offset;selected=selected[abs(signed)<=.01];signed=vertices[selected]@n+offset
        if len(selected)<200:break
        before=vertices[selected].copy();vertices[selected]-=signed[:,None]*n
        patches.append({'normal':n.tolist(),'offset':offset,'observed_vertex_indices':selected.tolist(),'rms_before_m':float(np.sqrt(np.mean(signed**2))), 'max_displacement_m':float(np.linalg.norm(vertices[selected]-before,axis=1).max())})
        remaining=remaining[~np.isin(remaining,selected)]
    result=dict(mesh);result['vertices']=vertices.tolist()
    result['plane_regularization']={'patches':patches,'max_allowed_displacement_m':.01,'added_faces':0,'removed_faces':0,'holes_preserved':True,'source':'observed vertices only, no state or GT'}
    result['source']+='; bounded observed plane regularization'
    return result


def silhouette_depth_sphere(depth, confidence, k, e, masks, scale, times):
    """Shared unknown radius; silhouette tangency plus confidence-selected front surface."""
    contours=[];surfaces=[];initial=[];radii=[];audit=[]
    for t,mask in enumerate(masks):
        cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if len(cs)!=1:raise ValueError('fragmented_silhouette')
        c=cs[0];area=cv2.contourArea(c);perimeter=cv2.arcLength(c,True)
        if 4*np.pi*area/max(perimeter**2,1)<.65:raise ValueError('non_sphere_silhouette')
        boundary=c[:,0].astype(float)
        # OpenCV contour traces boundary pixel centers: move half a pixel outward.
        center2=boundary.mean(axis=0);out=boundary-center2
        boundary+=.5*out/np.maximum(np.linalg.norm(out,axis=1)[:,None],1e-8)
        boundary=boundary[np.linspace(0,len(boundary)-1,min(256,len(boundary))).astype(int)]
        ray=np.c_[boundary,np.ones(len(boundary))]@np.linalg.inv(k[t]).T
        ray/=np.linalg.norm(ray,axis=1)[:,None];contours.append(ray)
        valid=(cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0)&np.isfinite(depth[t])&(depth[t]>0)&np.isfinite(confidence[t])
        if valid.sum()<20:raise ValueError('insufficient_depth_samples')
        valid &= confidence[t]>=np.median(confidence[t][valid])
        y,x=np.where(valid)
        choose=np.linspace(0,len(x)-1,min(256,len(x))).astype(int);x=x[choose];y=y[choose]
        xyz=(np.c_[x,y,np.ones(len(x))]@np.linalg.inv(k[t]).T)*depth[t,y,x,None]*scale
        surfaces.append(xyz)
        z=float(np.median(xyz[:,2]));theta=float(np.median(np.linalg.norm((boundary-center2)/[k[t,0,0],k[t,1,1]],axis=1)))
        r=z*theta/(1-theta)
        if not 0<theta<.8 or r<=0:raise ValueError('invalid_silhouette_initialization')
        initial.append((np.linalg.inv(k[t])@np.r_[center2,1])*(z+r));radii.append(r)
        audit.append({'contour_points':len(ray),'depth_points':len(x),'depth_confidence_median':float(np.median(confidence[t][valid]))})
    r0=float(np.median(radii))
    def residual(params):
        centers=params[:-1].reshape(8,3);radius=params[-1];out=[]
        for c,rays,xyz in zip(centers,contours,surfaces):
            tangent=np.linalg.norm(np.cross(rays,c),axis=1)-radius
            surface=np.linalg.norm(xyz-c,axis=1)-radius
            out.extend([tangent/r0/np.sqrt(len(tangent)),surface/r0/np.sqrt(len(surface))])
        return np.concatenate(out)
    lower=np.full(25,-np.inf);lower[-1]=1e-6
    # A visible front cap must precede its center. Prevent the mirror/back-cap
    # optimum during fitting rather than accepting a small algebraic residual.
    lower[2:24:3]=[np.percentile(xyz[:,2],90) for xyz in surfaces]
    x0=np.r_[np.array(initial).ravel(),r0]
    x0[2:24:3]=np.maximum(x0[2:24:3],lower[2:24:3]+.01*r0)
    fit=least_squares(residual,x0,bounds=(lower,np.full(25,np.inf)),loss='soft_l1',f_scale=.01,max_nfev=200)
    camera_centers=fit.x[:-1].reshape(8,3);radius=float(fit.x[-1])
    centers=np.stack([(c-e[t,:,3]*scale)@e[t,:,:3] for t,c in enumerate(camera_centers)])
    tangent_rms=float(np.sqrt(np.mean(np.concatenate([(np.linalg.norm(np.cross(ray,c),axis=1)-radius)/radius for ray,c in zip(contours,camera_centers)])**2)))
    surface_rms=float(np.sqrt(np.mean(np.concatenate([(np.linalg.norm(xyz-c,axis=1)-radius)/radius for xyz,c in zip(surfaces,camera_centers)])**2)))
    tt=np.array(times)-times[-1];design=np.c_[np.ones(8),tt]
    beta=np.linalg.lstsq(design,centers,rcond=None)[0]
    motion=least_squares(lambda x:(design@x.reshape(2,3)-centers).ravel(),beta.ravel(),loss='soft_l1',f_scale=radius*.05)
    beta=motion.x.reshape(2,3);motion_rms=float(np.sqrt(np.mean((design@beta-centers)**2)))
    reasons=[]
    if not fit.success:reasons.append('sphere_optimizer_failed')
    if tangent_rms>.1:reasons.append('silhouette_residual')
    if surface_rms>.15:reasons.append('surface_residual')
    if not motion.success:reasons.append('motion_optimizer_failed')
    if motion_rms>.25*radius:reasons.append('motion_residual')
    if any(np.median(xyz[:,2])>=c[2] for xyz,c in zip(surfaces,camera_centers)):reasons.append('surface_not_in_front_of_center')
    return {'status':'FAIL' if reasons else 'ESTIMATED','reasons':reasons,'shape':'sphere',
        'radius':radius,'center_sequence':centers.tolist(),'p7':beta[0].tolist(),'v7':beta[1].tolist(),
        'omega':[0,0,0],'omega_source':'explicit_zero_baseline','orientation':{'status':'UNKNOWN','collision_quaternion':[0,0,0,1]},
        'relative_rms':surface_rms,'silhouette_relative_rms':tangent_rms,'motion_rms_m':motion_rms,
        'radius_cv':float(np.std(radii)/np.mean(radii)),'frames':audit,
        'source':'silhouette tangency + confidence-selected visible depth; unknown radius; no support alignment'}
