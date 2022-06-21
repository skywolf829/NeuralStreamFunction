import torch
import torch.nn as nn
import torch.nn.functional as F
from Other.utility_functions import make_coord_grid

'''
Backwards is not implemented for pytorch's grid sample, so we use
this existing grid_sample_3d code from user DongJT1996 on Github.
https://github.com/pytorch/pytorch/issues/34704
'''
def grid_sample_3d(volume, points):
    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = points.shape

    ix = points[..., 0]
    iy = points[..., 1]
    iz = points[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)


    with torch.no_grad():

        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    volume = volume.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(volume, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(volume, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(volume, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(volume, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(volume, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(volume, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(volume, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(volume, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val

class Grid(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        grid_shape = [1, opt['n_outputs']]
        for _ in range(opt['n_dims']):
            grid_shape.append(opt['grid_size'])
        g = torch.ones(grid_shape, 
            dtype=torch.float32, 
            device=opt['device'])
        self.grid = nn.Parameter(data=g,requires_grad=True)

        with torch.no_grad():
            self.grid.uniform_(-1, 1)
    
    def forward(self, coords):     
        output = grid_sample_3d(self.grid, coords.unsqueeze(1).unsqueeze(1).unsqueeze(0))
        output = output.squeeze().unsqueeze(1)
        return output

    def forward_w_grad(self, coords):
        coords = coords.requires_grad_(True)
        output = self(coords)
        return output, coords
    
    def forward_maxpoints(self, coords, max_points=100000):
        output_shape = list(coords.shape)
        output_shape[-1] = self.opt['n_outputs']
        output = torch.empty(output_shape, 
            dtype=torch.float32, device=self.opt['device'])
        for start in range(0, coords.shape[0], max_points):
            #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
            output[start:min(start+max_points, coords.shape[0])] = \
                self(coords[start:min(start+max_points, coords.shape[0])])
        return output


    def sample_grid(self, grid, max_points = 100000):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid, max_points = 100000)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        return vals

    def sample_grad_grid(self, grid, 
        output_dim = 0, max_points=1000):
        
        coord_grid = make_coord_grid(grid, 
            self.opt['device'], False)
        
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)       

        output_shape = list(coord_grid.shape)
        output_shape[-1] = self.opt['n_dims']
        output = torch.empty(output_shape, 
            dtype=torch.float32, device=self.opt['device'], 
            requires_grad=False)

        for start in range(0, coord_grid.shape[0], max_points):
            vals = self(
                coord_grid[start:min(start+max_points, coord_grid.shape[0])])
            grad = torch.autograd.grad(vals[:,output_dim], 
                coord_grid, 
                grad_outputs=torch.ones_like(vals[:,output_dim])
                )[0][start:min(start+max_points, coord_grid.shape[0])]
            
            output[start:min(start+max_points, coord_grid.shape[0])] = grad

        output = output.reshape(coord_grid_shape)
        
        return output

    def sample_grid_for_image(self, grid, boundary_scaling = 1.0):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        if(len(coord_grid.shape) == 4):
            coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
        
        coord_grid *= boundary_scaling

        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        if(self.opt['loss'] == "l1occupancy"):
            vals = vals[..., 0:-1]
        return vals

    def sample_occupancy_grid_for_image(self, grid, boundary_scaling = 1.0):
        coord_grid = make_coord_grid(grid, self.opt['device'], False)
        if(len(coord_grid.shape) == 4):
            coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
        
        coord_grid *= boundary_scaling

        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
        vals = self.forward_maxpoints(coord_grid)
        coord_grid_shape[-1] = self.opt['n_outputs']
        vals = vals.reshape(coord_grid_shape)
        if(self.opt['loss'] == "l1occupancy"):
            vals = vals[...,-1]
        return vals
    
    def sample_grad_grid_for_image(self, grid, boundary_scaling = 1.0, 
        input_dim = 0, output_dim = 0):

        coord_grid = make_coord_grid(grid, self.opt['device'], False)        
        if(len(coord_grid.shape) == 4):
            coord_grid = coord_grid[:,:,int(coord_grid.shape[2]/2),:]
        coord_grid *= boundary_scaling
        
        coord_grid_shape = list(coord_grid.shape)
        coord_grid = coord_grid.view(-1, coord_grid.shape[-1]).requires_grad_(True)
        vals = self.forward_maxpoints(coord_grid)    


        grad = torch.autograd.grad(vals[:,output_dim], 
            coord_grid,#[:,input_dim], 
            grad_outputs=torch.ones_like(vals[:, output_dim]),
            allow_unused=True)
        

        grad = grad[0][:,input_dim]
        coord_grid_shape[-1] = 1
        grad = grad.reshape(coord_grid_shape)
        
        return grad
    
    def sample_rect(self, starts, widths, samples):
        positions = []
        for i in range(len(starts)):
            positions.append(
                torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                    dtype=torch.float32, device=self.opt['device'])
            )
        grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1)
        vals = self.forward(grid_to_sample)
        return vals

    def sample_grad_rect(self, starts, widths, samples, input_dim, output_dim):
        positions = []
        for i in range(len(starts)):
            positions.append(
                torch.arange(starts[i], starts[i] + widths[i], widths[i] / samples[i], 
                    dtype=torch.float32, device=self.opt['device'])
            )
        grid_to_sample = torch.stack(torch.meshgrid(*positions), dim=-1).requires_grad_(True)
        vals = self.forward(grid_to_sample)
        
        grad = torch.autograd.grad(vals[:,output_dim], 
            grid_to_sample,#[:,input_dim], 
            grad_outputs=torch.ones_like(vals[:, output_dim]),
            allow_unused=True)
        

        grad = grad[0][:,input_dim]
        grid_to_sample[-1] = 1
        grad = grad.reshape(grid_to_sample)

        return vals