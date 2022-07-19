# stripe-noise-removal
A simple demo to remove strip noise using Fourier transform:
transform the original image to Fourier domain, then select a shuttle region in the stripe direction to remove the frequency domain signal, and obtain the denoised image by inverse Fourier transform.
Run denoise.py for stripe removal
and the BM3D method can be used as further denoising
