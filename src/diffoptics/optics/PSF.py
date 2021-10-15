import torch


class PSF(torch.nn.Module):

    def __init__(self, kernel: torch.tensor):
        super(PSF, self).__init__()

        self.kernel = kernel

    def forward(self, x: torch.tensor):
        # Fourier convolution
        fft_x = torch.fft.fftn(x)
        fft_kernel = torch.fft.fftn(self.kernel.to(x.device), s=fft_x.shape)

        # Multiplication and inverse fourier transform
        output = torch.fft.ifftn(fft_x * fft_kernel)

        # Discard complex parts
        return output.real
