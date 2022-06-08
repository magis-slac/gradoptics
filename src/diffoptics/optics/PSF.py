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
        fft_kernel = torch.fft.fftn(self.kernel.to(x.device), s=fft_x.shape)

        # Multiplication and inverse fourier transform
        output = torch.fft.ifftn(fft_x * fft_kernel)

        # Discard complex parts
        return output.real
