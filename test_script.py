import PurePython
import OffAxisProjectionCythonized
import numpy as np
import time

def benchmark_rotation_matrix():
    total_a = 0.
    total_b = 0.
    iterations = 1000
    for i in range(iterations):
        start = time.time()
        a = PurePython.get_rotation_matrix(np.array([3., 6., 9.]))
        #print(PurePython.get_rotation_matrix(np.array([3., 6., 9.])))
        total_a += time.time() - start
        start = time.time()
        b = OffAxisProjectionCythonized.get_rotation_matrix(np.array([3., 6., 9.]))
        #print(OffAxisProjectionCythonized.get_rotation_matrix(np.array([3., 6., 9.])))
        total_b += time.time() - start
        assert np.allclose(a, b, rtol=1e-09)
    print(total_a / iterations)
    print(total_b / iterations)


def single_benchmark_rot():
    start = time.time()
    a = PurePython.get_rotation_matrix(np.array([3., 6., 9.]))
    #print(PurePython.get_rotation_matrix(np.array([3., 6., 9.])))
    print(time.time() - start)
    start = time.time()
    b = OffAxisProjectionCythonized.get_rotation_matrix(np.array([3., 6., 9.]))
    #print(OffAxisProjectionCythonized.get_rotation_matrix(np.array([3., 6., 9.])))
    print(time.time() - start)
    assert np.allclose(a, b, rtol=1e-09)


def single_benchmark_OffAP():
    num_particles = 10000

    # px and py contains randomly generated values between 0 and 1
    px = np.random.random(num_particles)
    py = np.random.random(num_particles)
    pz = np.random.random(num_particles)
    particle_masses = np.ones(num_particles)
    particle_densities = np.ones(num_particles)
    smoothing_length = np.random.random(num_particles)
    quantity_to_smooth = np.ones(num_particles)
    bounds = [-1, 1, -1, 1, -1, 1]
    normal_vector = np.array([-2., 2., -5])
    resolution = (512, 512)
    buf = np.zeros(resolution)

    start = time.time()
    PurePython.off_axis_projection_SPH(px, py, pz, particle_masses,
                                       particle_densities, smoothing_length,
                                       bounds, quantity_to_smooth, buf, normal_vector)
    print(time.time() - start)
    start = time.time()
    OffAxisProjectionCythonized.off_axis_projection_SPH(px, py, pz, particle_masses,
                                                        particle_densities, smoothing_length,
                                                        bounds, quantity_to_smooth, buf, normal_vector)
    print(time.time() - start)


if __name__ == '__main__':
    single_benchmark_OffAP()