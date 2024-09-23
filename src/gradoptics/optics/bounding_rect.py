from gradoptics.optics.ray import Rays
from gradoptics.optics import BaseOptics
import torch
import numpy as np

from scipy.spatial import transform


class BoundingRect(BaseOptics):

    def __init__(self, width=1e-3, height=1e-3, depth=1e-3, xc=0.2, yc=0.0, zc=0.0, 
                 roll=0., pitch=0., yaw=0.):
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def get_ray_intersection_(self, incident_rays, eps=1e-15):
        """
        @Todo
        :param incident_rays:
        :param eps:
        :return:
        """
        
        # Computes the intersection of the incident_ray with the cube
        orig_origins = incident_rays.origins
        orig_directions = incident_rays.directions
        device = incident_rays.origins.device

        # Instead of rotating the cube, apply the inverse rotation to the rays
        inv_rot_mat = transform.Rotation.from_euler('XYZ', [self.roll, self.pitch, self.yaw]).as_matrix().T
        inv_rot_mat = torch.tensor(inv_rot_mat).to(device)
        expanded_rot = torch.eye(4,dtype=torch.float64).to(device)
        expanded_rot[:3, :3] = inv_rot_mat
        
        translation_vec = torch.tensor([self.xc, self.yc, self.zc]).to(orig_origins.device)

        origins = torch.matmul(expanded_rot[:-1], 
                               torch.cat((orig_origins-translation_vec, torch.ones((orig_origins.shape[0],1), device=device)), dim=1)[:, :, np.newaxis]).squeeze(dim=-1)+translation_vec
        directions = torch.matmul(expanded_rot[:-1], 
                               torch.cat((orig_directions, torch.zeros((orig_directions.shape[0],1), device=device)), dim=1)[:, :, np.newaxis]).squeeze(dim=-1)
        
        directions = directions/torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))

        # Now that we've rotated, we're axis aligned. Set up the faces of the cube
        fx0 = self.xc - self.depth/2 
        fx1 = self.xc + self.depth/2
        
        fy0 = self.yc - self.width/2 
        fy1 = self.yc + self.width/2
        
        fz0 = self.zc - self.height/2 
        fz1 = self.zc + self.height/2
        
        # Get the time of intersection with the planes corresponding to each face
        tx0 = ((fx0 - origins[:, 0]) / directions[:, 0])[:, None]
        tx1 = ((fx1 - origins[:, 0]) / directions[:, 0])[:, None]
        
        ty0 = ((fy0 - origins[:, 1]) / directions[:, 1])[:, None]
        ty1 = ((fy1 - origins[:, 1]) / directions[:, 1])[:, None]
        
        tz0 = ((fz0 - origins[:, 2]) / directions[:, 2])[:, None]
        tz1 = ((fz1 - origins[:, 2]) / directions[:, 2])[:, None]
        
        # And the corresponding points of intersection
        pointx0 = origins + tx0*directions
        pointx1 = origins + tx1*directions
        
        pointy0 = origins + ty0*directions
        pointy1 = origins + ty1*directions
        
        pointz0 = origins + tz0*directions
        pointz1 = origins + tz1*directions
        
        # Package everything up in a 6 face x n rays x 1 (3) tensor
        all_ts = torch.stack([tx0, tx1, 
                              ty0, ty1,
                              tz0, tz1])
            
        all_int = torch.stack([pointx0, pointx1, 
                              pointy0, pointy1, 
                              pointz0, pointz1])
        
        # Select the points that fall within the rect volume (with some tolerance)
        in_rect_mask = ((all_int[:, :, 0] >= fx0-eps) & (all_int[:, :, 0] <= fx1+eps) & 
                        (all_int[:, :, 1] >= fy0-eps) & (all_int[:, :, 1] <= fy1+eps) &
                        (all_int[:, :, 2] >= fz0-eps) & (all_int[:, :, 2] <= fz1+eps))

        all_t_min = all_ts.clone()
        all_t_min[~in_rect_mask] = float('inf')
        t_min = all_t_min.min(dim=0).values.squeeze()
        t_min[t_min.isinf()] = float('nan')
        
        all_t_max = all_ts.clone()
        all_t_max[~in_rect_mask] = -float('inf')
        t_max = all_t_max.max(dim=0).values.squeeze()
        t_max[t_max.isinf()] = float('nan')
       
        # Return intersections
        return t_min, t_max
    
    def get_ray_intersection(self, incident_rays, eps=1e-15):
        return self.get_ray_intersection_(incident_rays, eps=eps)[0]
    
    def intersect(self, incident_rays, t):
        origins = incident_rays.origins
        directions = incident_rays.directions

        # Update the origin of the incoming rays
        origins = origins + t.unsqueeze(1) * directions

        return Rays(origins,
                    directions,
                    luminosities=incident_rays.luminosities,
                    device=incident_rays.device,
                    meta=incident_rays.meta)
 
    def plot(self, ax, color='grey', alpha=0.4):
        phi = np.arange(1,10,2)*np.pi/4
        Phi, Theta = np.meshgrid(phi, phi)

        x = np.cos(Phi)*np.sin(Theta)
        y = np.sin(Phi)*np.sin(Theta)
        z = np.cos(Theta)/np.sqrt(2)
        
        ax.plot_surface(x*self.depth+self.xc, 
                        y*self.width+self.yc, 
                        z*self.height+self.zc, color=color, alpha=alpha)
