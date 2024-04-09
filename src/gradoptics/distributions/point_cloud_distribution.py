"""
Author: Suyong Choi (Department of Physics, Korea U.)
Date: 4/9/2024

Implements point atom source cloud to be used in gradoptics
"""

import torch
from gradoptics.distributions.base_distribution import BaseDistribution

# following is a utility class. The main class to be used, PointCloudDistribution, is below
class PointCloudVoxel(object):
    def __init__(self, point_cloud: torch.Tensor, nbins: int, device: str, morph: bool):
        """Initializer

        Args:
            point_cloud (torch.Tensor Nx3): 3D coordinates of N points
            nbins (int): number of bins per axis to bin the space
            device (str): 'cuda' or 'cpu' device
            morph (bool): whether to apply morphing of space to have similar number of entries per bin. 
                          Set to True for Gaussian distributed
        """
        self.point_cloud = point_cloud
        self.dtype = point_cloud.dtype
        self.npoints = point_cloud.shape[0]
        self.morph = morph
        self.min_coords = torch.min(self.point_cloud, dim=0).values
        self.max_coords = torch.max(self.point_cloud, dim=0).values + 1.0e-6 # add tiny value s.t. no atoms at the end
        self.min_morphx = self.morphcoordsyst(self.min_coords)
        self.max_morphx = self.morphcoordsyst(self.max_coords)

        self.nbins = nbins
        self.device = device
        
        self.build_voxels()
        self.x = 1.0e6*torch.ones((1,3), dtype=self.dtype, device=device)
        pass

    def morphcoordsyst(self, x:torch.Tensor) -> torch.Tensor:
        """ utility function to map real space coordinates to morphed coordinates using
            error function such that if the space has constant binning, each bin
            contains similar number of entries for a Gaussian distributed sample

        Args:
            x (torch.Tensor Nx3): Input coordinates

        Returns:
            torch.Tensor Nx3: Morphed coordinates
        """
        if self.morph:
            # scaled ranges from -3.0 to 3.0
            xscaled = (2.0*(x - self.min_coords) / (self.max_coords - self.min_coords) -1.0) * 3.0
            # use error function mapping s.t. each bin has similar number of entries
            morphx = torch.erf(xscaled)
        else:
            morphx = x
        return morphx


    def create_voxel_grid_morph(self)-> tuple[torch.Tensor, torch.Tensor]:
        """ Creates a voxel grid from a point cloud.

        Args:
            point_cloud (torch.Tensor): N x 3 tensor representing the point cloud.
            nbins (int): number of bins per dimension

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            torch.Tensor: Voxel grid with points assigned to voxels.
        """
        self.voxel_size = (self.max_morphx - self.min_morphx) / self.nbins # make slightly larger to avoid problem with points at the end


        # Assign points to voxels
        self.voxel_indices = ((self.morphcoordsyst(self.point_cloud) - self.min_morphx) / self.voxel_size).to(torch.int64)
        
        return self.voxel_indices, self.voxel_size

    def build_voxels(self):
        """Creates voxels for partitioning space
        """        
        self.create_voxel_grid_morph()

        # store index of atoms in voxel
        atoms_in_voxel =  [ [] for _ in range(self.nbins**3) ]

        # convert index to one dimensional
        linindextch = self.voxel_indices[:, 0] + self.nbins * self.voxel_indices[:, 1] + self.nbins*self.nbins * self.voxel_indices[:, 2]

        for iatom in range(self.npoints):
            #linindex = self.voxel_indices[iatom, 0] + self.nbins * self.voxel_indices[iatom, 1] + self.nbins*self.nbins * self.voxel_indices[iatom, 2]
            atoms_in_voxel[linindextch[iatom]].append(iatom)
        
        # Create a torch tensor that has the indices of atoms in column
        # for a voxel in rows. Since voxels have different number of atoms,
        # need to pad the empty columns with some number -1 in order
        # to create the torch tensor which only works with regular matrices
        # This matrix is used later to query the atoms associated with a voxel quickly.

        #Get the maximum length of any list in the list of lists
        max_length = max(len(l) for l in atoms_in_voxel)

        # Create a new list of lists, where each list is padded with zeros to the maximum length
        padded_list_of_lists = []
        for l in atoms_in_voxel:
            padded_list = l + [-1] * (max_length - len(l))
            padded_list_of_lists.append(padded_list)

        # Convert the padded list of lists to a tensor
        self.points_in_voxel = torch.tensor(padded_list_of_lists, dtype=torch.int64, device=self.device)
        self.npoints_in_voxel = (self.points_in_voxel+1).count_nonzero(dim=1) # keep account of how many valid points are in voxels
        # Add a special dummy atom 
        self.point_cloud = torch.cat([torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64, device=self.device), self.point_cloud])
        pass

    def get_voxels_closeto(self, x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get info on voxels close to position x

        Args:
            x (torch.Tensor Nx3): points in space

        Returns:
            tuple[torch.Tensor Nx3, torch.Tensor Nx1, torch.Tensor Nx1]: 3D indices of voxel x should be in, whether the indices are in bounds, linearized 1D index
        """        
        x_voxel_indices = ((self.morphcoordsyst(x) - self.min_morphx) / self.voxel_size).to(torch.int64)
        validvoxelhit = torch.all((x_voxel_indices >=0) & (x_voxel_indices < self.nbins), dim=1)
        linindex = x_voxel_indices[:,0] + self.nbins * x_voxel_indices[:,1] + self.nbins*self.nbins * x_voxel_indices[:,2]
        return x_voxel_indices, validvoxelhit, linindex
    
    def get_d2_voxels_around(self, x:torch.Tensor, i:int, j:int, k:int) -> tuple[torch.Tensor,torch.Tensor]:
        """_summary_

        Args:
            x (torch.Tensor Nx3): _description_
            i (int): index to be added to the x index of the voxel x should be in
            j (int): index to be added to the y index of the voxel x should be in
            k (int): index to be added to the z index of the voxel x should be in

        Returns:
            tuple[torch.Tensor Nx1, torch.Tensor Nx1]: distance squared of atoms in voxel to x, valid mask
        """
        if not torch.equal(self.x, x):
            self.x_voxel_indices, self.validvoxelhit, linindex = self.get_voxels_closeto(x)
            self.x = x

        maskgood = torch.zeros_like(self.validvoxelhit, device=self.device)
        d2 = 10.0 *torch.ones_like(self.validvoxelhit).unsqueeze(1).unsqueeze(2)

        if self.validvoxelhit.sum()>0:
            #atomsnearx = self.atoms_in_voxel[linindex[validvoxelhit]] + 1
            # include atoms in neighboring voxels
            xvshiftidx =  self.x_voxel_indices[self.validvoxelhit] + torch.tensor([i,j,k], dtype=torch.int64, device=self.device)
            xvshiftidxval = torch.all((xvshiftidx >=0) & (xvshiftidx < self.nbins), dim=1) # check valid index
            xvshiftlinindex =xvshiftidx[:,0] + self.nbins * xvshiftidx[:,1] + self.nbins*self.nbins * xvshiftidx[:,2]

            # get atom indices for the valid voxel
            if xvshiftidxval.sum()>0:
                atomsnearx = self.points_in_voxel[xvshiftlinindex[xvshiftidxval]]  + 1
                #atomsnearx = self.point_cloud[xvshiftlinindex[xvshiftidxval][:, :self.npoints_in_voxel[xvshiftlinindex[xvshiftidxval]]] ]  + 1
                if atomsnearx.shape[0]>0:
                    d2 = torch.square(self.point_cloud[atomsnearx] - x[self.validvoxelhit][xvshiftidxval].unsqueeze(1))
                    maskgood = torch.zeros_like(self.validvoxelhit, device=self.device).scatter(0, self.validvoxelhit.nonzero()[:,0], xvshiftidxval)
                    return d2, maskgood
        
        return d2, maskgood


class PointCloudDistribution(BaseDistribution):
    """Discrete atoms to be used in gradoptics as light source

    Args:
        BaseDistribution (_type_): _description_
    """    
    def __init__(self, points:torch.Tensor, sigma=2e-5, nbins=15, device='cuda'):
        """initializer

        Args:
            points (torch.Tensor): Atom positions N by 3 matrix.
            sigma (double, optional): Gaussian width. Defaults to 2e-5.
            nbins (int, optional): number of divisions per axis for voxelizing. Defaults to 15.
            device (str, optional): torch compute device. Defaults to 'cuda'.
        """        
        self.atoms = points
        self.sigma = sigma
        self.nbins = nbins
        self.device = device
        self.natoms = points.shape[0]
        self.pointcloudvoxel = PointCloudVoxel(points, nbins, self.device, morph=True)
        pass
    
    
    def pdf(self, x:torch.Tensor)->torch.Tensor:
        """Calculate PDF

        Args:
            x (torch.Tensor N by 3): positions where PDF is to be computed.

        Returns:
            torch.Tensor N by 1
        """        
        intensity = torch.zeros(x.shape[0], dtype=torch.float64, device=self.device)
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    d2, maskgood = self.pointcloudvoxel.get_d2_voxels_around(x, i, j, k)                    
                    if maskgood.sum()>0:
                        intensity[maskgood] += torch.sum(torch.exp(-torch.sum(d2, dim=2) / (2.0*self.sigma**2)), dim=1)

        return intensity
    
    def plot(self, ax, **kwargs):
        copytocpu = self.atoms.cpu().numpy()
        ax.scatter(copytocpu[:,0], copytocpu[:,1], copytocpu[:,2])
        pass

    def sample(self, nb_points:int, device='cpu')->torch.Tensor:
        """Sample atom positions

        Args:
            nb_points (int): number of samples
            device (str, optional): _description_. Defaults to 'cpu'.

        Returns:
            torch.Tensor: Position of nb_points atoms
        """        
        idx = torch.randint(1, self.natoms+1, (nb_points, ), device=device) # 0th entry contains dummy atom, so skip that
        selatoms = self.atoms[idx,:]
        # use the following to use sample as smeared position
        #modpos = selatoms + self.sigma*torch.randn_like(selatoms, device=device) 
        return selatoms 
    pass

# following should take a few seconds
if __name__=='__main__':
    atoms = 0.0005*torch.randn((100000, 3), device='cuda')
    cloud0 = PointCloudDistribution(atoms, sigma=5e-6)
    x = torch.tensor([[0., 0., 0.]], device='cuda')
    pdf = cloud0.pdf(x)
    voxelindices, valid, linindex = cloud0.pointcloudvoxel.get_voxels_closeto(x)
    print(f'{torch.count_nonzero(cloud0.pointcloudvoxel.points_in_voxel[linindex[valid]]+1, 1)} atoms in the voxel')
    print(f'intensity {pdf}')
    pass