# -*- coding: utf-8 -*-

"""raytracing.py

Implements basic ray tracing functionality
"""

import itertools
import multiprocessing
import numba
import numpy as np


class Raytracing(object):
    def __init__(self, camera, target, angle, resolution, plane_d=0.1, mask_sun=None):
        """Initialize raytracing class

        Parameters
        ----------
        camera_position : np.ndarray
            Camera position.
        target : [np.ndarray
            Camera target.
        viewport_angle : np.ndarray
            Viewport angle in degrees.
        resolution : tuple
            Image resolution (nx, ny).
        """
        self.angle = np.pi * angle / 180
        self.nx, self.ny = resolution

        self.camera_pos = camera
        self.target_pos = target
        self.sun_pos = mask_sun

        w = np.array([0, 0, -1])

        t = target - camera
        b = np.cross(w, t)

        t_n = t / np.linalg.norm(t)
        b_n = b / np.linalg.norm(b)
        v_n = np.cross(t_n, b_n)

        g_x = plane_d * np.tan(self.angle / 2)
        g_y = g_x * (self.nx - 1) / (self.ny - 1)

        self.q_x = 2 * g_x / (self.ny - 1) * b_n
        self.q_y = 2 * g_y / (self.nx - 1) * v_n
        self.p_1m = t_n * plane_d - g_x * b_n - g_y * v_n

    def generate_image(self, model, iparams, t_launch, t_image, density, **kwargs):
        """Generate raytraced image of the 3DCORE model at a given datetime.

        Parameters
        ----------
        model : Base3DCOREModel
            3DCORE model class.
        iparams : np.ndarray
            Inital parameters for the 3DCORE model.
        t_launch : datetime.datetime
            3DCORE launch datetime.
        t_image : datetime.datetime
            Datetime at which to generate the RT image.
        density : func
            Density function for the 3DCORE model in (q)-coordinates.

        Returns
        -------
        np.ndarray
            RT image.
        """
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        step_params = (kwargs.get("step_large", 0.05), kwargs.get("step_small", .0005),
                       kwargs.get("step_min", 0), kwargs.get("step_max", 2))

        rt_params = (self.camera_pos, self.sun_pos, self.p_1m, self.q_x, self.q_y)

        coords = itertools.product(list(range(self.nx)), list(range(self.ny)))

        result = pool.starmap(worker_generate_image,
                              [(model, i, j, iparams, t_launch, t_image, step_params, rt_params,
                                density)
                               for (i, j) in coords])

        image = np.zeros((self.nx, self.ny))

        for pix in result:
            image[pix[0], pix[1]] = pix[2]

        return image


def worker_generate_image(model, i, j, iparams, t_launch, t_image, step_params, rt_params, density):
    if i % 16 == 0 and j == 0:
        print("tracing line", i)
    step_large, step_small, step_min, step_max = step_params
    e, s, p, q_x, q_y = rt_params

    iparams_arr = iparams.reshape(1, -1)

    model_obj = model(t_launch, runs=1, use_gpu=False)
    model_obj.update_iparams(iparams_arr, seed=42)
    model_obj.propagate(t_image)

    rho_0, rho_1 = model_obj.sparams_arr[0, 2:4]

    # large steps
    step_large_cnt = int((step_max - step_min) // step_large)
    rays_large = np.empty((step_large_cnt, 3), dtype=np.float32)
    t_steps_large = np.linspace(step_min, step_max, step_large_cnt)

    rt_rays(rays_large, t_steps_large, i, j, rt_params)

    rays_large_qs = np.empty((len(rays_large), 3), dtype=np.float32)

    model_obj.transform_sq(rays_large, rays_large_qs)

    # detect if ray is close to 3DCORE structure or the origin (sun)
    mask = generate_mask(model_obj, i, j, rays_large, rays_large_qs, step_large, step_large_cnt,
                         rho_0, rho_1)

    if mask[0] == -1:
        return i, j, np.nan
    elif np.all(mask == 0):
        return i, j, 0

    arg_first = np.argmax(mask > 0) - 1
    arg_last = len(mask) - np.argmax(mask[::-1] > 0)

    if arg_last == len(mask):
        arg_last -= 1

    # small steps
    step_small_cnt = int((t_steps_large[arg_last] - t_steps_large[arg_first]) // step_small)

    rays_small = np.empty((step_small_cnt, 3), dtype=np.float32)
    t_steps_small = np.linspace(t_steps_large[arg_first], t_steps_large[arg_last], step_small_cnt)

    rt_rays(rays_small, t_steps_small, i, j, rt_params)

    rays_small_qs = np.empty((len(rays_small), 3), dtype=np.float32)

    model_obj.transform_sq(rays_small, rays_small_qs)

    intensity = integrate_intensity(i, j, rays_small, rays_small_qs, step_small_cnt, rho_0, rho_1,
                                    e, density)

    return i, j, intensity


@numba.njit(inline="always")
def rt_p_ij(i, j, p, q_x, q_y):
    return p + q_x * (i - 1) + q_y * (j - 1)


@numba.njit(inline="always")
def rt_ray(e, i, j, t, p, q_x, q_y):
    ray = rt_p_ij(i, j, p, q_x, q_y)
    ray = ray / np.linalg.norm(ray)

    return e + t * ray


@numba.njit(parallel=True)
def rt_rays(rays_arr, t_steps, i, j, rt_params):
    e, _, p, q_x, q_y = rt_params

    for k in numba.prange(len(t_steps)):
        rays_arr[k] = rt_ray(e, i, j, t_steps[k], p, q_x, q_y)

    return rays_arr


def generate_mask(model_obj, i, j, rays_large, rays_large_qs, step_large, step_large_cnt, rho_0,
                  rho_1):
    mask = np.zeros((len(rays_large)), dtype=np.float32)

    fail_flag = False

    for k in range(step_large_cnt):
        q0, q1, q2 = rays_large_qs[k]

        # check distance to sun
        if np.linalg.norm(rays_large[k]) < 2 * step_large:
            mask[k] = 1

            if np.linalg.norm(rays_large[k]) < 0.00465047:
                fail_flag = True
                break

    # compute rays surface points
    rays_large_qs_surf = np.array(rays_large_qs)
    rays_large_qs_surf[:, 0] = 1

    rays_large_surf = np.empty_like(rays_large_qs_surf)

    model_obj.transform_qs(rays_large_surf, rays_large_qs_surf)

    for k in range(step_large_cnt):
        # check distance to cme
        if np.linalg.norm(rays_large[k] - rays_large_surf[k]) < 2 * step_large:
            mask[k] = 1

    if fail_flag:
        return -1 * np.ones_like(mask)
    else:
        return mask


@numba.njit
def integrate_intensity(i, j, rays_small, rays_small_qs, step_small_cnt, rho_0, rho_1, e, density):
    intensity = 0

    fail_flag = False

    for k in range(step_small_cnt):
        q0, q1, q2 = rays_small_qs[k]

        # check distance to sun
        if np.linalg.norm(rays_small[k]) < 0.00465047:
            fail_flag = True
            break

        # check distance to cme
        if q0 < 3:
            v1 = e - rays_small[k]
            v2 = rays_small[k]
            thomson_cos_sq = np.dot(v1, v2)**2 / np.linalg.norm(v1)**2 * np.linalg.norm(v2)**2
            intensity += thomson_cos_sq * density(q0, q1, q2, rho_0, rho_1)

    if fail_flag:
        return np.nan
    else:
        return intensity
