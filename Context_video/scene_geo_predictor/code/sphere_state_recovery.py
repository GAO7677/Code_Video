"""Observation-only known-radius sphere fitting; no scene or GT dependencies."""
import numpy as np
from scipy.optimize import least_squares
from skimage.measure import find_contours


def fit_rays(pixels, intrinsic, radius):
    if radius <= 0 or len(pixels) < 20:
        raise ValueError('invalid explicit radius or insufficient contour')
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsic).T
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    axis = rays.mean(axis=0)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(rays @ axis, -1, 1))
    initial = axis * radius / np.sin(np.median(angle))
    def residual(center):
        return np.linalg.norm(np.cross(rays, center), axis=1) - radius
    result = least_squares(residual, initial, loss='soft_l1', f_scale=.002,
                           bounds=([-np.inf, -np.inf, radius], [np.inf]*3), max_nfev=500)
    if not result.success or np.any(rays @ result.x <= 0):
        raise ValueError('sphere tangent optimization failed')
    residuals = residual(result.x)
    return result.x, {'residual_rms_m': float(np.sqrt(np.mean(residuals**2))),
                      'inlier_ratio': float(np.mean(np.abs(residuals) <= .005)),
                      'contour_points': len(pixels), 'residuals_m': residuals.tolist()}


def fit_mask(mask, intrinsic, extrinsic, radius):
    if mask.ndim != 2 or not mask.any() or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        raise ValueError('empty or image-clipped silhouette')
    contours = find_contours(mask.astype(float), .5)
    # Keep the complete observed boundary, including multiple components; no circle substitution.
    pixels = np.concatenate(contours)[:, ::-1]
    center, report = fit_rays(pixels, intrinsic, radius)
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    report['component_count'] = len(contours)
    mean = pixels.mean(axis=0)
    pixel_radius = np.median(np.linalg.norm(pixels-mean, axis=1))
    z = np.sqrt(intrinsic[0,0]*intrinsic[1,1])*radius/pixel_radius
    legacy = np.linalg.inv(intrinsic) @ np.r_[mean, 1.] * z
    return rotation.T @ (center-translation), rotation.T @ (legacy-translation), report


def motion_fits(centers, times):
    t = np.asarray(times)-times[-1]
    if len(t) != 8 or not np.all(np.diff(t)>0):
        raise ValueError('require eight increasing observation timestamps')
    fits = {'last2': {'p7': centers[-1].tolist(), 'v7': ((centers[-1]-centers[-2])/(t[-1]-t[-2])).tolist()}}
    for degree, name in [(1,'robust_linear'), (2,'robust_quadratic')]:
        design = np.column_stack([t**i for i in range(degree+1)])
        initial = np.linalg.lstsq(design, centers, rcond=None)[0]
        fit = least_squares(lambda x: (design @ x.reshape(degree+1,3)-centers).ravel(),
                            initial.ravel(), loss='soft_l1', f_scale=.01)
        if not fit.success:
            raise ValueError('motion fit failed')
        coeff = fit.x.reshape(degree+1,3)
        fits[name] = {'p7': coeff[0].tolist(), 'v7': coeff[1].tolist(),
                      'observation_rms_m': float(np.sqrt(np.mean((design@coeff-centers)**2)))}
    return fits
