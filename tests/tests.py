import torch
import numpy as np
import gradoptics as optics
import matplotlib.pyplot as plt
from gradoptics.optics import batch_vector
from gradoptics.optics.vector import cross_product
from gradoptics.optics.bounding_sphere import BoundingSphere
from gradoptics.transforms.look_at_transform import LookAtTransform
from gradoptics.transforms.simple_transform import SimpleTransform


def _test_ray(dim=1000):
    o = batch_vector(torch.zeros(dim), torch.zeros(dim), torch.zeros(dim))
    d = batch_vector(torch.ones(dim), torch.zeros(dim), torch.zeros(dim))
    ray = optics.Rays(o, d)

    if ((ray.origins.shape[0] == ray.directions.shape[0] == dim) and (
            ray.origins.shape[1] == ray.directions.shape[1] == 3)):
        return 0
    else:
        return -1


def _test_rejection_sampling():
    # Atom cloud
    atom_cloud = optics.AtomCloud()

    # Define a sampler to sample from the cloud density
    proposal_dist = optics.GaussianDistribution(mean=0., std=0.0002)
    x = optics.rejection_sampling(atom_cloud.marginal_cloud_density_x, int(1e6), proposal_dist, m=None)
    y = optics.rejection_sampling(atom_cloud.marginal_cloud_density_y, int(1e6), proposal_dist, m=None)
    z = optics.rejection_sampling(atom_cloud.marginal_cloud_density_z, int(1e6), proposal_dist, m=None)

    plt.hist(x.numpy(), bins=300, histtype='step', color='k', label='x')
    plt.hist(y.numpy(), bins=300, histtype='step', color='C0', label='y')
    plt.hist(z.numpy(), bins=300, histtype='step', color='b', label='z')
    plt.legend()
    plt.savefig('test_rejection_sampling.pdf')
    plt.close()

    return 0


def _test_atom_cloud(nb_atoms=int(1e4)):
    # Atom cloud
    atom_cloud = optics.LightSourceFromDistribution(optics.AtomCloud())

    rays = atom_cloud.sample_rays(nb_atoms)

    origins = rays.origins
    directions = rays.directions

    if not ((origins.shape[0] == directions.shape[0] == nb_atoms) and (origins.shape[1] == directions.shape[1] == 3)):
        return -1

    # Plot the origin of the rays (the cloud)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], alpha=.3, s=0.5)
    ax.view_init(azim=0, elev=0)
    ax.set_zlim([origins[:, 2].mean() - 3 * origins[:, 2].std(), origins[:, 2].mean() + 3 * origins[:, 2].std()])
    ax.set_ylim([origins[:, 1].mean() - 3 * origins[:, 1].std(), origins[:, 1].mean() + 3 * origins[:, 1].std()])
    ax.set_xlim([origins[:, 0].mean() - 3 * origins[:, 0].std(), origins[:, 0].mean() + 3 * origins[:, 0].std()])
    plt.savefig('test_atom_cloud.pdf')
    plt.close()

    return 0


def _test_lens(nb_rays=50, f=0.05, m=0.15, right_of_lens=True, position=[0., 0., 0.]):
    lens = optics.PerfectLens(f=f, m=m, position=position)

    # Create rays parallel to the optical axis
    z_pos = torch.linspace(-lens.f * lens.na / 2 + 1e-5, lens.f * lens.na / 2 - 1e-5, nb_rays)
    origins = batch_vector(torch.zeros(nb_rays) + (1000 if right_of_lens else -1000),
                           torch.zeros(nb_rays) + position[1],
                           z_pos + position[2])

    directions = batch_vector(torch.ones(nb_rays) * (-1 if right_of_lens else 1),
                              torch.zeros(nb_rays),
                              torch.zeros(nb_rays))

    rays = optics.Rays(origins, directions)
    t = lens.get_ray_intersection(rays)
    refracted_rays, _ = lens.intersect(rays, t)

    # Computes the time t at which the rays will intersect with the image focal plane
    o = refracted_rays.origins
    d = refracted_rays.directions
    f_ = f if right_of_lens else -f
    t = (o[:, 0] - position[0] - f_) / d[:, 0]

    pt = torch.empty((nb_rays, 3))
    pt[:, 0] = o[:, 0] + t * d[:, 0]
    pt[:, 1] = o[:, 1] + t * d[:, 1]
    pt[:, 2] = o[:, 2] + t * d[:, 2]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    # Plot the incoming rays
    for i in range(o.shape[0]):
        ax.plot([o[i, 0] - directions[i, 0] / 10, o[i, 0]],
                [o[i, 1], o[i, 1]],
                [o[i, 2], o[i, 2]])
    # Plot the rays refracted by the lens
    for i in range(o.shape[0]):
        ax.plot([o[i, 0], pt[i, 0]],
                [o[i, 1], pt[i, 1]],
                [o[i, 2], pt[i, 2]])
    plt.savefig('test_lens.pdf')
    plt.close()

    # Check that all rays intersect at the same point
    pt_mean = pt.mean(dim=0)
    error = 0
    for point in pt:
        error += ((point - pt_mean) ** 2).sum().sqrt()

    if (error / nb_rays) < 5e-3:
        return 0
    else:
        return -1


def _test_lens_transform(nb_rays=50, f=0.05, m=0.15, right_of_lens=True, position=[0., 0., 0.]):
    transform = SimpleTransform(0., 45 / 180 * np.pi, 0., position)

    lens = optics.PerfectLens(f=f, m=m, position=position, transform=transform)

    # Create rays parallel to the optical axis
    z_pos = torch.linspace(-lens.f * lens.na / 2 + 1e-5, lens.f * lens.na / 2 - 1e-5, nb_rays)
    origins = batch_vector(torch.zeros(nb_rays) + (1000 if right_of_lens else -1000),
                           torch.zeros(nb_rays) + position[1],
                           z_pos + position[2])

    directions = batch_vector(torch.ones(nb_rays) * (-1 if right_of_lens else 1),
                              torch.zeros(nb_rays),
                              torch.zeros(nb_rays))

    rays = optics.Rays(origins, directions)
    t = lens.get_ray_intersection(rays)
    refracted_rays, _ = lens.intersect(rays, t)

    # Computes the time t at which the rays will intersect with the image focal plane
    o = refracted_rays.origins
    d = refracted_rays.directions
    f_ = f if right_of_lens else -f
    t = (o[:, 0] - position[0] - f_) / d[:, 0]

    pt = torch.empty((nb_rays, 3))
    pt[:, 0] = o[:, 0] + t * d[:, 0]
    pt[:, 1] = o[:, 1] + t * d[:, 1]
    pt[:, 2] = o[:, 2] + t * d[:, 2]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    # Plot the incoming rays
    for i in range(o.shape[0]):
        ax.plot([o[i, 0] - directions[i, 0] / 10, o[i, 0]],
                [o[i, 1], o[i, 1]],
                [o[i, 2], o[i, 2]])
    # Plot the rays refracted by the lens
    for i in range(o.shape[0]):
        ax.plot([o[i, 0], pt[i, 0]],
                [o[i, 1], pt[i, 1]],
                [o[i, 2], pt[i, 2]])
    plt.savefig('test_lens_transform.pdf')
    plt.close()

    return 0


def _test_thick_lens(nb_rays=64):

    position = torch.tensor([0., 0., 0.])
    transform = optics.simple_transform.SimpleTransform(0, 0, 0, position)
    lens = optics.ThickLens(1.5, 1., 1, 1e-1, transform)

    # Create rays parallel to the optical axis
    y_pos = torch.linspace(-.1, .1, int(np.sqrt(nb_rays)))
    z_pos = torch.linspace(-.1, .1, int(np.sqrt(nb_rays)))
    y_pos, z_pos = torch.meshgrid((y_pos, z_pos))

    origins = optics.batch_vector(torch.zeros(nb_rays) + 2, y_pos + position[1], z_pos + position[2])
    directions = optics.batch_vector(torch.ones(nb_rays) * -1, torch.zeros(nb_rays), torch.zeros(nb_rays))
    rays = optics.Rays(origins, directions)

    t = lens.get_ray_intersection(rays)
    mask = ~torch.isnan(t)
    refracted_rays, _ = lens.intersect(rays[mask], t[mask])

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    rays[mask].plot(ax, t[mask])
    refracted_rays.plot(ax, [1.2 for _ in range(refracted_rays.origins.shape[0])], color='k', linestyle='--')
    ax.view_init(elev=30., azim=70)
    plt.savefig('test_thick_lens.pdf')
    plt.close()

    return 0


def _ray_marching_test_sensor(eps=1e-3):
    f = 0.05
    m = 0.15

    sensor = optics.Sensor(position=(-f * (1 + m), 0, 0))
    pts = sensor.sample_points_on_sensor(1000000)

    plt.hist2d(pts[:, 1].numpy(), pts[:, 2].numpy(), bins=40)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig('test_ray_marching_sensor.pdf')
    plt.close()

    pts[:, 1] /= sensor.pixel_size[1]
    pts[:, 2] /= sensor.pixel_size[0]

    if ((pts[:, 1].max() - eps) < sensor.resolution[1] / 2) and \
            ((pts[:, 1].min() + eps) > -sensor.resolution[1] / 2) and \
            ((pts[:, 2].max() - eps) < sensor.resolution[0] / 2) and \
            ((pts[:, 2].min() + eps) > -sensor.resolution[0] / 2):
        return 0
    else:
        return -1


def _ray_marching_test_lens(eps=1e-3):
    f = 0.05
    na = 1 / 1.4

    lens = optics.PerfectLens(f=f, na=na)
    pts = lens.sample_points_on_lens(1000000)

    plt.figure()
    plt.hist2d(pts[:, 1].numpy(), pts[:, 2].numpy(), bins=40)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.savefig('test_ray_marching_lens.pdf')
    plt.close()

    if (pts.max() <= (f * na / 2 + eps)) and (pts.min() >= (-f * na / 2 - eps)):
        return 0
    else:
        return -1


def _test_window():
    obj_x_pos = 0.31
    left_interface_x_position = obj_x_pos + .056
    right_interface_x_position = left_interface_x_position + .05
    window = optics.Window(left_interface_x_position, right_interface_x_position)
    atom_cloud = optics.LightSourceFromDistribution(optics.AtomCloud(position=[obj_x_pos, 0., 0.]))
    nb_atoms = int(1e5)

    rays = atom_cloud.sample_rays(nb_atoms)
    t = window.get_ray_intersection(rays)
    r = rays[~torch.isnan(t)]
    t = t[~torch.isnan(t)]
    rays, _ = window.intersect(r, t)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    n = 100
    window.plot(ax)
    rays.plot(ax, [.1 for _ in range(n)], color='k')
    r.plot(ax, t[:n])
    plt.savefig('test_window.pdf')
    plt.close()

    # The refracted direction should be the same as the incident direction
    if ((rays.directions[:n] - r.directions[:n]).abs().sum() / n) < 1e-5:
        return 0
    else:
        return -1


def _test_cross_product():
    assert (cross_product(torch.tensor([1, 0, 0]), torch.tensor([0, 1, 0])) == torch.tensor([0, 0, 1])).all()
    assert (cross_product(torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])) == torch.tensor([1, 0, 0])).all()
    assert (cross_product(torch.tensor([0, 0, 1]), torch.tensor([1, 0, 0])) == torch.tensor([0, 1, 0])).all()
    assert (cross_product(torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0])) == torch.tensor([0, 0, -1])).all()
    assert (cross_product(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])) == torch.tensor([-1, 0, 0])).all()
    assert (cross_product(torch.tensor([1, 0, 0]), torch.tensor([0, 0, 1])) == torch.tensor([0, -1, 0])).all()
    return 0


def _test_look_at_transform():
    up = torch.tensor([0., 0., 1.])
    viewing_direction = torch.tensor([1., 0., 0.])
    pos = torch.tensor([0., 0., 0.])
    transform = LookAtTransform(viewing_direction, pos, up=up)

    assert (transform.transform == torch.tensor([[0., 0., 1., 0.],
                                                 [1., 0., 0., 0.],
                                                 [0., 1., 0., 0.],
                                                 [0., 0., 0., 1.]])).all()

    up = torch.tensor([0., 0., 1.])
    viewing_direction = torch.tensor([-1., 0., 0.])
    pos = torch.tensor([0., 0., 0.])
    transform = LookAtTransform(viewing_direction, pos, up)

    assert (transform.transform == (torch.tensor([[0., 0., -1., 0.],
                                                  [-1., 0., 0., 0.],
                                                  [0., 1., 0., 0.],
                                                  [0., 0., 0., 1.]]))).all()

    up = torch.tensor([1., 0., 0.])
    viewing_direction = torch.tensor([0., 0., 1.])
    pos = torch.tensor([0., 0., 0.])
    transform = LookAtTransform(viewing_direction, pos, up)

    assert (transform.transform == (torch.tensor([[0., 1., 0., 0.],
                                                  [-1., 0., 0., 0.],
                                                  [0., -0., 1., 0.],
                                                  [0., 0., 0., 1.]]))).all()

    up = torch.tensor([-1., 0., 0.])
    viewing_direction = torch.tensor([0., 0., 1.])
    pos = torch.tensor([0., 0., 0.])
    transform = LookAtTransform(viewing_direction, pos, up)

    assert (transform.transform == (torch.tensor([[0., -1., 0., 0.],
                                                  [1., 0., 0., 0.],
                                                  [-0., 0., 1., 0.],
                                                  [0., 0., 0., 1.]]))).all()
    return 0


def _test_grad_rays():
    def toy_intersect(rays: optics.Rays, normal=torch.tensor([.2, .2, .6])) -> optics.Rays:
        x = (rays.directions[:, 0] * normal[0]).reshape(-1, 1)
        y = (rays.directions[:, 1] * normal[1]).reshape(-1, 1)
        z = (rays.directions[:, 2] * normal[2]).reshape(-1, 1)
        new_directions = torch.cat((x, y, z), dim=1)
        new_origins = rays.origins * 2 + 1
        return optics.Rays(new_origins, new_directions)

    origins = torch.randn(1, 3, requires_grad=True)
    directions = torch.randn(1, 3, requires_grad=True)
    rays1 = optics.Rays(origins, directions)
    rays2 = toy_intersect(rays1)
    assert directions.is_leaf
    assert origins.is_leaf

    toy_loss = (rays2.origins * rays2.directions).sum()
    toy_loss.backward()
    assert origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


def _test_bounding_sphere(nb_rays=1000):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')

    cloud_envelopes = [BoundingSphere(xc=3e-3, radii=1e-3),
                       BoundingSphere(xc=-3e-3, radii=1e-3),
                       BoundingSphere(xc=0, yc=3e-3, radii=1e-3),
                       BoundingSphere(xc=0, yc=-3e-3, radii=1e-3)]

    for cloud_envelope in cloud_envelopes:
        cloud_envelope.plot(ax)

    # Rays
    origins = torch.zeros((nb_rays, 3))
    # Sample rays in 4 pi
    azimuthal_angle = torch.rand(nb_rays) * 2 * np.pi
    polar_angle = torch.arccos(1 - 2 * torch.rand(nb_rays))
    emitted_direction = optics.batch_vector(torch.sin(polar_angle) * torch.sin(azimuthal_angle),
                                            torch.sin(polar_angle) * torch.cos(azimuthal_angle),
                                            torch.cos(polar_angle))
    rays = optics.Rays(origins, emitted_direction)

    # First intersection with the sphere
    t_s = []
    for cloud_envelope in cloud_envelopes:
        t_s.append(cloud_envelope.get_ray_intersection(rays))

    for t in t_s:
        rays.plot(ax, t)
        assert (t[~torch.isnan(t)] > 0).all()  # Check that all the times are greater than 0

    # Second intersection with the sphere
    for i, cloud_envelope in enumerate(cloud_envelopes):
        cond = ~torch.isnan(t_s[i])
        outgoing_rays, _ = cloud_envelope.intersect(rays[cond], t_s[i][cond])
        t_ = cloud_envelope.get_ray_intersection(outgoing_rays)
        outgoing_rays.plot(ax, t_, color='r')
        cond_ = torch.isnan(t_)
        assert cond_.sum() == 0  # In this setup, All the rays should make it to the second intersection
        assert torch.allclose(rays[cond].directions,
                              outgoing_rays.directions)  # The sphere should not modify the directions
        outgoing_rays, _ = cloud_envelope.intersect(outgoing_rays, t_)
        assert torch.allclose(rays[cond].directions,
                              outgoing_rays.directions)  # The sphere should not modify the directions

    ax.set_xlim([-0.004, 0.004])
    ax.set_ylim([-0.004, 0.004])
    ax.set_zlim([-0.004, 0.004])
    plt.savefig('test_bounding_sphere.pdf')
    plt.close()
    return 0


def _test_grad_mirror_wrt_incident_rays(mirror_position=torch.ones(3),
                                        ray_origins=torch.randn(2, 3, requires_grad=True)):
    mirror = optics.FlatMirror(mirror_position[0], mirror_position[1], mirror_position[2], torch.tensor([.2, .2, .6]),
                               .005)

    directions = [(mirror_position[0] - ray_origins[:, 0]).reshape(-1, 1),
                  (mirror_position[1] - ray_origins[:, 1]).reshape(-1, 1),
                  (mirror_position[2] - ray_origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    directions = directions.clone().detach().requires_grad_(True)
    rays = optics.Rays(ray_origins, directions)
    t = mirror.get_ray_intersection(rays)

    outgoing_rays, _ = mirror.intersect(rays, t)

    loss = (outgoing_rays.origins * outgoing_rays.directions).sum()
    loss.backward()

    assert ray_origins.is_leaf
    assert directions.is_leaf
    assert ray_origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


def _test_grad_lens_wrt_incident_rays(lens_position=[1., 1., 1.], ray_origins=torch.randn(2, 3, requires_grad=True)):
    lens = optics.PerfectLens(position=lens_position)
    directions = [(lens_position[0] - ray_origins[:, 0]).reshape(-1, 1),
                  (lens_position[1] - ray_origins[:, 1]).reshape(-1, 1),
                  (lens_position[2] - ray_origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    directions = directions.clone().detach().requires_grad_(True)
    rays = optics.Rays(ray_origins, directions)
    t = lens.get_ray_intersection(rays)

    outgoing_rays, _ = lens.intersect(rays, t)

    loss = (outgoing_rays.origins * outgoing_rays.directions).sum()
    loss.backward()

    assert directions.is_leaf
    assert ray_origins.is_leaf
    assert ray_origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0


"""def _test_grad_sensor_wrt_incident_rays(sensor_position=torch.ones(3),
                                        ray_origins=torch.randn(2, 3, requires_grad=True)):
    sensor = optics.Sensor(position=sensor_position)
    directions = [(sensor_position[0] - ray_origins[:, 0]).reshape(-1, 1),
                  (sensor_position[1] - ray_origins[:, 1]).reshape(-1, 1),
                  (sensor_position[2] - ray_origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    directions = directions.clone().detach().requires_grad_(True)
    rays = optics.Rays(ray_origins, directions)
    t = sensor.get_ray_intersection(rays)

    hit_position, _, _ = sensor.intersect(rays, t)

    target_position = torch.randn(hit_position.shape)

    l1_loss = ((hit_position - target_position) ** 2).mean()
    l1_loss.backward()

    assert directions.is_leaf
    assert ray_origins.is_leaf
    assert ray_origins.grad.abs().sum() > 0
    assert directions.grad.abs().sum() > 0
    return 0
"""


def _test_grad_mirror_wrt_self_parameters():
    mirror_normal = torch.tensor([.2, .2, .6], requires_grad=True)
    mirror_position = torch.tensor([1., 1., 1.], requires_grad=True)
    mirror = optics.FlatMirror(mirror_position[0], mirror_position[1], mirror_position[2], mirror_normal, .005)

    origins = torch.randn(2, 3)
    directions = [(mirror_position[0].item() - origins[:, 0]).reshape(-1, 1),
                  (mirror_position[1].item() - origins[:, 1]).reshape(-1, 1),
                  (mirror_position[2].item() - origins[:, 2]).reshape(-1, 1)]
    directions = torch.cat(directions, dim=1)
    rays = optics.Rays(origins, directions)
    t = mirror.get_ray_intersection(rays)
    outgoing_rays, _ = mirror.intersect(rays, t)

    # Toy loss
    loss = (outgoing_rays.origins * outgoing_rays.directions).sum()
    loss.backward()

    assert mirror_normal.is_leaf
    assert mirror_position.is_leaf
    assert mirror_position.grad.abs().sum() > 0
    assert mirror_normal.grad.abs().sum() > 0
    return 0


def _test_grad_lens_wrt_self_parameters():
    return 0


def _test_grad_sensor_wrt_self_parameters():
    return 0


def _test_forward_ray_tracing(f=0.05, m=0.15, device='cpu'):
    # Creating a scene
    image_pof = -f * (1 + m)
    object_pof = f * (1 + m) / m
    lens = optics.PerfectLens(f=f, na=1 / 1.4, position=[0., 0., 0.], m=m)
    sensor = optics.Sensor(resolution=(9600, 9600), pixel_size=(3.76e-6, 3.76e-6), position=(image_pof, 0, 0),
                           poisson_noise_mean=2, quantum_efficiency=0.8)
    atom_cloud = optics.AtomCloud(n=int(1e6), f=2, position=[object_pof, 0., 0.], phi=0.1)
    light_source = optics.LightSourceFromDistribution(atom_cloud)
    scene = optics.Scene(light_source)
    scene.add_object(lens)
    scene.add_object(sensor)

    # Using the built-in function forward_ray_tracing
    rays = light_source.sample_rays(10_000_000, device=device)
    optics.forward_ray_tracing(rays, scene, max_iterations=2)

    # Readout the sensor
    produced_image = sensor.readout(add_poisson_noise=False).data.cpu().numpy()

    c = (4800, 4800)
    w = 40
    plt.imshow(produced_image[c[0] - w: c[0] + w, c[1] - w: c[1] + w], cmap='Blues')
    plt.savefig('test_forward_ray_tracing.pdf')
    plt.close()

    return 0


def _test_backward_ray_tracing(f=0.05, m=0.15, device='cpu'):
    # Creating a scene
    image_pof = -f * (1 + m)
    object_pof = f * (1 + m) / m
    lens = optics.PerfectLens(f=f, na=1 / 1.4, position=[0., 0., 0.], m=m)
    sensor = optics.Sensor(resolution=(9600, 9600), pixel_size=(3.76e-6, 3.76e-6), position=(image_pof, 0, 0),
                           poisson_noise_mean=2, quantum_efficiency=0.8)
    atom_cloud = optics.AtomCloud(n=int(1e6), f=2, position=[object_pof, 0., 0.], phi=0.1)
    light_source_bounding_shape = optics.BoundingSphere(radii=1e-3, xc=f * (1 + m) / m, yc=0.0, zc=0.0)
    light_source = optics.LightSourceFromDistribution(atom_cloud, bounding_shape=light_source_bounding_shape)
    scene = optics.Scene(light_source)
    scene.add_object(lens)
    scene.add_object(sensor)

    # Computing incident rays (assuming pinhole camera)
    N = 40
    px_j, px_i = torch.meshgrid(torch.linspace(N, -N + 1, steps=N * 2), torch.linspace(N, -N + 1, steps=N * 2))
    px_j = px_j.reshape(-1, 1).type(torch.long)
    px_i = px_i.reshape(-1, 1).type(torch.long)
    pos_x = (px_i - 0.5) * sensor.pixel_size[0]
    pos_y = (px_j - 0.5) * sensor.pixel_size[1]
    pos_z = torch.zeros(pos_x.shape)
    origins = torch.cat((pos_x, pos_y, pos_z), dim=1)
    origins = sensor.c2w.apply_transform_(origins)
    directions = optics.batch_vector(- origins[:, 0], - origins[:, 1], - origins[:, 2])
    incident_rays = optics.Rays(origins, directions, device=device)

    # Producing an image with backward ray tracing
    integrator = optics.StratifiedSamplingIntegrator(100)
    image = optics.backward_ray_tracing(incident_rays, scene, light_source, integrator, max_iterations=2)
    image = image.reshape(2 * N, 2 * N).data.cpu().numpy()

    plt.imshow(image)
    plt.savefig('test_backward_ray_tracing.pdf')
    plt.close()

    return 0


def _test_curved_mirrors():
    # Creating a scene
    f = 0.05
    m = 0.15
    lens = optics.PerfectLens(f=f, na=1 / 1.4, position=[0., 0., 0.], m=m)
    sensor = optics.Sensor(position=(-f * (1 + m), 0, 0))
    atom_cloud = optics.AtomCloud(n=int(1e6), f=2, position=[f * (1 + m) / m / 2, 0., 0.], phi=0.1)
    mirror = optics.CurvedMirror(40, .05, .05, optics.simple_transform.SimpleTransform(
        0., 0., 0., torch.tensor([f * (1 + m) / m * 3 / 4, 0, 0])))
    light_source = optics.LightSourceFromDistribution(atom_cloud)
    scene = optics.Scene(light_source)
    scene.add_object(lens)
    scene.add_object(sensor)
    scene.add_object(mirror)

    # Producing an image
    device = 'cpu'
    rays = light_source.sample_rays(10_00_000, device=device)
    rays.directions[:, 0] = rays.directions[:, 0].abs()
    optics.forward_ray_tracing(rays, scene, max_iterations=3)

    # Readout the sensor
    c = (4800, 4800)
    w = 40
    produced_image = sensor.readout(add_poisson_noise=False).data.cpu().numpy()
    plt.imshow(produced_image[c[0] - w: c[0] + w, c[1] - w: c[1] + w], cmap='Blues')
    plt.savefig('test_curved_mirrors.pdf')
    plt.close()
    return 0


"""
            Tests for pytest
"""


def test_ray():
    assert _test_ray(1) == 0
    assert _test_ray(1000) == 0


def test_rejection_sampling():
    assert _test_rejection_sampling() == 0


def test_atom_cloud():
    assert _test_atom_cloud() == 0


def test_lens():
    for lens_position in [[0., 0., 0.], [-0.1, 0., 0.], [0.1, 0.1, 0.1]]:
        assert _test_lens(f=0.05, m=0.15, position=lens_position) == 0
        assert _test_lens(f=0.04, m=0.15, position=lens_position) == 0
        assert _test_lens(f=0.01, m=0.15, position=lens_position) == 0
        assert _test_lens(f=0.05, m=0.1, position=lens_position) == 0
        assert _test_lens(f=0.04, m=0.1, position=lens_position) == 0
        assert _test_lens(f=0.01, m=0.1, position=lens_position) == 0

        assert _test_lens(f=0.05, m=0.15, right_of_lens=False, position=lens_position) == 0
        assert _test_lens(f=0.01, m=0.15, right_of_lens=False, position=lens_position) == 0
        assert _test_lens(f=0.05, m=0.1, right_of_lens=False, position=lens_position) == 0

    assert _test_lens_transform(f=0.05, m=0.15, position=[0., 0., 0.]) == 0


def test_thick_lens():
    assert _test_thick_lens() == 0


def test_ray_marching():
    assert _ray_marching_test_sensor() == 0
    assert _ray_marching_test_lens() == 0


def test_window():
    assert _test_window() == 0


def test_gradients_wrt_incident_rays():
    """
    For each optical component, check that the gradients with respect the incident rays propagate correctly
    """
    assert _test_grad_rays() == 0
    assert _test_grad_mirror_wrt_incident_rays() == 0
    assert _test_grad_lens_wrt_incident_rays() == 0
    # assert _test_grad_sensor_wrt_incident_rays() == 0


def test_gradients_wrt_self_parameters():
    """
    For each optical component, check that the gradients with respect its parameters propagate correctly
    """
    assert _test_grad_mirror_wrt_self_parameters() == 0
    assert _test_grad_lens_wrt_self_parameters() == 0
    assert _test_grad_sensor_wrt_self_parameters() == 0


def test_vectors():
    assert _test_cross_product() == 0


def test_transforms():
    assert _test_look_at_transform() == 0


def test_bounding_sphere():
    assert _test_bounding_sphere() == 0


def test_forward_ray_tracing():
    assert _test_forward_ray_tracing() == 0


def test_backward_ray_tracing():
    assert _test_backward_ray_tracing() == 0


def test_curved_mirrors():
    assert _test_curved_mirrors() == 0
