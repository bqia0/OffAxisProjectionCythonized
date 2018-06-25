import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def off_axis_projection_SPH(np.float64_t[:] px, 
                            np.float64_t[:] py, 
                            np.float64_t[:] pz, 
                            np.float64_t[:] particle_masses, 
                            np.float64_t[:] particle_densities,
                            np.float64_t[:] smoothing_lengths, 
                            bounds, 
                            np.float64_t[:] quantity_to_smooth,
                            np.float64_t[:, :] projection_array, 
                            np.float64_t[:] normal_vector):
    # Do nothing in event of a 0 normal vector
    if np.allclose(normal_vector, np.array([0., 0., 0.]), rtol=1e-09):
        return

    cdef int num_particles = min(np.size(px), np.size(py), np.size(pz),
                        np.size(particle_masses))
    cdef np.float64_t[:, :] rotation_matrix = get_rotation_matrix(normal_vector)
    cdef np.float64_t[:] px_rotated = np.empty(num_particles, dtype='float_')
    cdef np.float64_t[:] py_rotated = np.empty(num_particles, dtype='float_')
    cdef np.float64_t x_coordinate
    cdef np.float64_t y_coordinate
    cdef np.float64_t z_coordinate
    cdef np.float64_t[:] coordinate_matrix
    cdef np.float64_t[:] rotated_coordinates
    cdef np.float64_t bounds_x0 = bounds[0]
    cdef np.float64_t bounds_x1 = bounds[1]
    cdef np.float64_t bounds_y0 = bounds[2]
    cdef np.float64_t bounds_y1 = bounds[3]
    cdef np.float64_t bounds_z0 = bounds[4]
    cdef np.float64_t bounds_z1 = bounds[5] 

    for i in range(num_particles):
        x_coordinate = px[i]
        y_coordinate = py[i]
        z_coordinate = pz[i]
        if x_coordinate > bounds_x1 or y_coordinate > bounds_y1:
            continue
        if x_coordinate < bounds_x0 or y_coordinate < bounds_y0:
            continue
        if z_coordinate < bounds_z0 or z_coordinate > bounds_z1:
            continue
        coordinate_matrix = np.array([x_coordinate, y_coordinate,
                                      z_coordinate], dtype='float_')
        rotated_coordinates = np.matmul(rotation_matrix, coordinate_matrix)
        if rotated_coordinates[0] < bounds_x0 or \
            rotated_coordinates[0] >= bounds_x1:
            continue
        if rotated_coordinates[1] < bounds_y0 or \
            rotated_coordinates[1] >= bounds_y1:
            continue
        px_rotated[i] = rotated_coordinates[0]
        py_rotated[i] = rotated_coordinates[1]
        
    # pixelize_sph_kernel_projection(projection_array,
    #                                px_rotated,
    #                                py_rotated,
    #                                smoothing_lengths,
    #                                particle_masses,
    #                                particle_densities,
    #                                quantity_to_smooth,
    #                                bounds[:4])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t[:, :] get_rotation_matrix(np.float64_t[:] normal_vector):
    """ Returns a numpy rotation matrix corresponding to the
    rotation of the z-axis ([0, 0, 1]) to a given normal vector
    https://math.stackexchange.com/a/476311
    """

    cdef np.float64_t[:] z_axis = np.array([0., 0., 1.], dtype='float_')
    cdef np.float64_t[:] normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
    cdef np.float64_t[:] v = np.cross(z_axis, normal_unit_vector)
    cdef np.float64_t s = np.linalg.norm(v)
    cdef np.float64_t c = np.dot(z_axis, normal_unit_vector)
    # if the normal vector is identical to the z-axis, just return the
    # identity matrix
    if np.isclose(c, 1, rtol=1e-09):
        return np.identity(3, dtype='float_')
    # if the normal vector is the negative z-axis, return zero matrix
    if np.isclose(s, 0, rtol=1e-09):
        return np.zeros((3, 3), dtype='float_')

    cdef np.float64_t[:, :] cross_product_matrix = np.array([[0, -1 * v[2], v[1]],
                                                      [v[2], 0, -1 * v[0]],
                                                      [-1 * v[1], v[0], 0]], 
                                                      dtype='float_')
    return np.identity(3, dtype='float_') + cross_product_matrix \
        + np.matmul(cross_product_matrix, cross_product_matrix) \
        * 1/(1+c)