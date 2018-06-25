import numpy as np


def off_axis_projection_SPH(px, py, pz, particle_masses, particle_densities,
                            smoothing_lengths, bounds, quantity_to_smooth,
                            projection_array, normal_vector):
    # Do nothing in event of a 0 normal vector
    if np.allclose(normal_vector, np.array([0., 0., 0.]), rtol=1e-09):
        return

    num_particles = min(np.size(px), np.size(py), np.size(pz),
                        np.size(particle_masses))
    rotation_matrix = get_rotation_matrix(normal_vector)

    # Allocate space for rotated coordinates
    px_rotated = np.zeros(num_particles, dtype='float_')
    py_rotated = np.zeros(num_particles, dtype='float_')

    for i in range(num_particles):
        x_coordinate = px[i]
        y_coordinate = py[i]
        z_coordinate = pz[i]
        if x_coordinate > bounds[1] or y_coordinate > bounds[3]:
            continue
        if x_coordinate < bounds[0] or y_coordinate < bounds[2]:
            continue
        if z_coordinate < bounds[4] or z_coordinate > bounds[5]:
            continue
        coordinate_matrix = np.array([x_coordinate, y_coordinate,
                                      z_coordinate], dtype='float_')
        rotated_coordinates = rotation_matrix @ coordinate_matrix
        if rotated_coordinates[0] < bounds[0] or \
            rotated_coordinates[0] >= bounds[1]:
            continue
        if rotated_coordinates[1] < bounds[2] or \
            rotated_coordinates[1] >= bounds[3]:
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


def get_rotation_matrix(normal_vector):
    """ Returns a numpy rotation matrix corresponding to the
    rotation of the z-axis ([0, 0, 1]) to a given normal vector
    https://math.stackexchange.com/a/476311
    """

    z_axis = np.array([0., 0., 1.], dtype='float_')
    normal_unit_vector = normal_vector / np.linalg.norm(normal_vector)
    v = np.cross(z_axis, normal_unit_vector)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, normal_unit_vector)
    # if the normal vector is identical to the z-axis, just return the
    # identity matrix
    if np.isclose(c, 1, rtol=1e-09):
        return np.identity(3, dtype='float_')
    # if the normal vector is the negative z-axis, return zero matrix
    if np.isclose(s, 0, rtol=1e-09):
        return np.zeros((3, 3), dtype='float_')

    cross_product_matrix = np.array([[0, -1 * v[2], v[1]],
                                    [v[2], 0, -1 * v[0]],
                                    [-1 * v[1], v[0], 0]], dtype='float_')
    return np.identity(3, dtype='float_') + cross_product_matrix \
        + cross_product_matrix @ cross_product_matrix \
        * 1/(1+c)