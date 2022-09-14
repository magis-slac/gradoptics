import torch


class PSF(torch.nn.Module):
    """
    Models a point spread function.
    """

    def __init__(self, kernel):
        """
        :param kernel: Kernel that characterizes the point spread function (:obj:`torch.tensor`)
        """
        super(PSF, self).__init__()
        self.kernel = kernel

    def forward(self, x):
        """
        Performs a convolution between the input data and the kernel of the point spread function

        :param x: Data to convolve with the PSF (:obj:`torch.tensor`)

        :return: The convolved input (:obj:`torch.tensor`)
        """
        # Fourier convolution
        fft_x = torch.fft.fftn(x)

        # Pad with zeros
        sz = (x.shape[0] - self.kernel.shape[0], x.shape[1] - self.kernel.shape[1])
        kernel_pad = torch.nn.functional.pad(self.kernel, ((sz[1]+1)//2, sz[1]//2, (sz[0]+1)//2, sz[0]//2))

        # Adjust coordinates to have center of kernel at top left
        kernel_shift = torch.fft.ifftshift(kernel_pad)

        # Then transform into Fourier space
        fft_kernel = torch.fft.fftn(kernel_shift.to(x.device))

        # Multiplication and inverse fourier transform
        output = torch.fft.ifftn(fft_x * fft_kernel)

        # Discard complex parts
        return output.real
