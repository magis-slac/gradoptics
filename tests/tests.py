import torch
import matplotlib.pyplot as plt

from diffoptics.inference.RejectionSampling import rejection_sampling
from diffoptics.optics import batch_vector, Rays, PerfectLens, Window, Sensor
from diffoptics.distributions.GaussianDistribution import GaussianDistribution
from diffoptics.light_sources.AtomCloud import AtomCloud


def _test_ray(dim=1000):
    o = batch_vector(torch.zeros(dim), torch.zeros(dim), torch.zeros(dim))
    d = batch_vector(torch.ones(dim), torch.zeros(dim), torch.zeros(dim))
    ray = Rays(o, d)

    if ((ray.origins.shape[0] == ray.directions.shape[0] == dim) and (
            ray.origins.shape[1] == ray.directions.shape[1] == 3)):
        return 0
    else:
        return -1


def _test_rejection_sampling():
    # Atom cloud
    atom_cloud = AtomCloud()

    # Define a sampler to sample from the cloud density
    proposal_dist = GaussianDistribution(mean=0., std=0.0002)
    x = rejection_sampling(atom_cloud.marginal_cloud_density_x, int(1e6), proposal_dist, m=None)
    y = rejection_sampling(atom_cloud.marginal_cloud_density_y, int(1e6), proposal_dist, m=None)
    z = rejection_sampling(atom_cloud.marginal_cloud_density_z, int(1e6), proposal_dist, m=None)

    plt.hist(x.numpy(), bins=300, histtype='step', color='k', label='x')
    plt.hist(y.numpy(), bins=300, histtype='step', color='C0', label='y')
    plt.hist(z.numpy(), bins=300, histtype='step', color='b', label='z')
    plt.legend()
    plt.savefig('test_rejection_sampling.pdf')
    plt.close()

    return 0


def _test_atom_cloud(nb_atoms=int(1e4)):
    # Atom cloud
    atom_cloud = AtomCloud()

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


def _test_lens(nb_rays=50, f=0.05, m=0.15, right_of_lens=True):
    lens = PerfectLens(f=f, m=m)

    # Create rays parallel to the optical axis
    z_pos = torch.linspace(-lens.f * lens.na / 2 + 1e-5, lens.f * lens.na / 2 - 1e-5, nb_rays)
    origins = batch_vector(torch.zeros(nb_rays) + (1000 if right_of_lens else -1000),
                           torch.zeros(nb_rays),
                           z_pos)

    directions = batch_vector(torch.ones(nb_rays) * (-1 if right_of_lens else 1),
                              torch.zeros(nb_rays),
                              torch.zeros(nb_rays))

    rays = Rays(origins, directions)
    t = lens.get_ray_intersection(rays)
    refracted_rays = lens.intersect(rays, t)

    # check if all rays intersect at the focal length
    o = refracted_rays.origins
    d = refracted_rays.directions
    f_ = f if right_of_lens else -f
    t = (o[:, 0] - f_) / d[:, 0]

    pt = torch.empty((nb_rays, 3))
    pt[:, 0] = o[:, 0] + t * d[:, 0]
    pt[:, 1] = o[:, 1] + t * d[:, 1]
    pt[:, 2] = o[:, 2] + t * d[:, 2]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    for i in range(o.shape[0]):
        ax.plot([o[i, 0] - directions[i, 0] / 10, o[i, 0]],
                [o[i, 1], o[i, 1]],
                [o[i, 2], o[i, 2]])
    for i in range(o.shape[0]):
        ax.plot([o[i, 0], pt[i, 0]],
                [o[i, 1], pt[i, 1]],
                [o[i, 2], pt[i, 2]])
    plt.savefig('test_lens.pdf')
    plt.close()

    pt_mean = pt.mean(dim=0)
    error = 0
    for point in pt:
        error += ((point - pt_mean) ** 2).sum().sqrt()

    if (error / nb_rays) < 5e-3:
        return 0
    else:
        return -1


def _ray_marching_test_sensor(eps=1e-3):
    f = 0.05
    m = 0.15

    sensor = Sensor(position=(-f * (1 + m), 0, 0))
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

    lens = PerfectLens(f=f, na=na)
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
    window = Window(left_interface_x_position,
                    right_interface_x_position)
    atom_cloud = AtomCloud(position=torch.tensor([obj_x_pos, 0., 0.]))
    nb_atoms = int(1e5)

    rays = atom_cloud.sample_rays(nb_atoms)
    t = window.get_ray_intersection(rays)
    r = rays.get_at(~torch.isnan(t))
    t = t[~torch.isnan(t)]
    rays = window.intersect(r, t)

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
    assert _test_lens(f=0.05, m=0.15) == 0
    assert _test_lens(f=0.04, m=0.15) == 0
    assert _test_lens(f=0.01, m=0.15) == 0
    assert _test_lens(f=0.05, m=0.1) == 0
    assert _test_lens(f=0.04, m=0.1) == 0
    assert _test_lens(f=0.01, m=0.1) == 0

    assert _test_lens(f=0.05, m=0.15, right_of_lens=False) == 0
    assert _test_lens(f=0.01, m=0.15, right_of_lens=False) == 0
    assert _test_lens(f=0.05, m=0.1, right_of_lens=False) == 0


def test_ray_marching():
    assert _ray_marching_test_sensor() == 0
    assert _ray_marching_test_lens() == 0


def test_window():
    assert _test_window() == 0