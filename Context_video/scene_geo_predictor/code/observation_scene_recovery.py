"""Observation-only recovery mechanisms; no dataset names or GT imports."""
import cv2
import numpy as np
from scipy.optimize import least_squares


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
    lower=np.full(25,-np.inf);lower[2:24:3]=1e-6;lower[-1]=1e-6
    fit=least_squares(residual,np.r_[np.array(initial).ravel(),r0],bounds=(lower,np.full(25,np.inf)),loss='soft_l1',f_scale=.01,max_nfev=200)
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
