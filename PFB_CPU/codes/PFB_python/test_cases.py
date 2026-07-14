#!/usr/bin/env python3

import numpy as np
from ipdb import set_trace as stop
import test_signals as ts
import PFB
import OverPFB
import matplotlib.pyplot as plt
import scipy
import matplotlib.animation as animation

def test_temporal_dirac_comb(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=False, real=True, is_complex=False, waterfall=True, temporal_plot=False, overPFB=False):
    """ Test temporal resolution in a PFB spectrometer by generating a dirac comb signal and analyzing the resulting spectrum. """
    osr = 32/27 # Oversampling ratio
    data = ts.generate_dirac_comb_signal(n_taps, n_chan, n_windows, delta_period, delta_start, include_noise=include_noise, real=real, is_complex=is_complex)
    # Apply PFB spectrometer to the generated dirac comb signal
    X_PFB = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming", PSD=False)
    x_PFB_psd = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming", PSD=True)
    x_fft = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="rectangular", PSD=False)
    x_fft_psd = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="rectangular", PSD=True)
    x_PFB_over = OverPFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, osr=osr, n_int=1, window_fn="hamming", PSD=False)
    x_PFB_over_psd = OverPFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, osr=osr, n_int=1, window_fn="hamming", PSD=True)

    # 3. Convert PSD outputs to dB (Power)
    x_PFB_psd_db = PFB.db(x_PFB_psd)
    x_fft_psd_db = PFB.db(x_fft_psd)
    x_PFB_over_psd_db = PFB.db(x_PFB_over_psd)

    # 4. Extract Real and Imaginary components for raw outputs (Amplitude)
    X_PFB_real = np.real(X_PFB)
    X_PFB_imag = np.imag(X_PFB)
    x_fft_real = np.real(x_fft)
    x_fft_imag = np.imag(x_fft)
    x_PFB_over_real = np.real(x_PFB_over)
    x_PFB_over_imag = np.imag(x_PFB_over)

    if overPFB:
        fig, axs = plt.subplots(3, 3, figsize=(18, 18))
        plot_kwargs = {
            'aspect': 'auto', 
            'origin': 'lower', 
            'cmap': 'viridis', 
            'interpolation': 'nearest'
        }
        im00 = axs[0, 0].imshow(x_PFB_over_real, **plot_kwargs)
        axs[0, 0].set_title("Over PFB Real Output")
        axs[0, 0].set_xlabel("Channel")
        axs[0, 0].set_ylabel("Time [Blocks]")
        fig.colorbar(im00, ax=axs[0, 0], label="Amplitude")

        im01 = axs[1, 0].imshow(x_PFB_over_imag, **plot_kwargs)
        axs[1, 0].set_title("Over PFB Imaginary Output")
        axs[1, 0].set_xlabel("Channel")
        axs[1, 0].set_ylabel("Time [Blocks]")
        fig.colorbar(im01, ax=axs[1, 0], label="Amplitude")

        im02 = axs[2, 0].imshow(x_PFB_over_psd_db, **plot_kwargs)
        axs[2, 0].set_title("Over PFB PSD Output [dB]")
        axs[2, 0].set_xlabel("Channel")
        axs[2, 0].set_ylabel("Time [Blocks]")
        fig.colorbar(im02, ax=axs[2, 0], label="Power [dB]")

        im10 = axs[0, 1].imshow(X_PFB_real, **plot_kwargs)
        axs[0, 1].set_title("PFB real Output [dB]")
        axs[0, 1].set_xlabel("Channel")
        axs[0, 1].set_ylabel("Time [Blocks]")
        fig.colorbar(im10, ax=axs[0, 1], label="Power [dB]")

        im11 = axs[1, 1].imshow(X_PFB_imag, **plot_kwargs)
        axs[1, 1].set_title("PFB Imaginary Output [dB]")
        axs[1, 1].set_xlabel("Channel")
        axs[1, 1].set_ylabel("Time [Blocks]")
        fig.colorbar(im11, ax=axs[1, 1], label="Power [dB]")

        im12 = axs[2, 1].imshow(x_PFB_psd_db, **plot_kwargs)
        axs[2, 1].set_title("PFB PSD Output [dB]")
        axs[2, 1].set_xlabel("Channel")
        axs[2, 1].set_ylabel("Time [Blocks]")
        fig.colorbar(im12, ax=axs[2, 1], label="Power [dB]")

        im20 = axs[0, 2].imshow(x_fft_real, **plot_kwargs)
        axs[0, 2].set_title("Standard FFT Real Output [dB]")
        axs[0, 2].set_xlabel("Channel")
        axs[0, 2].set_ylabel("Time [Blocks]")
        fig.colorbar(im20, ax=axs[0, 2], label="Power [dB]")

        im21 = axs[1, 2].imshow(x_fft_imag, **plot_kwargs)
        axs[1, 2].set_title("Standard FFT Imaginary Output [dB]")
        axs[1, 2].set_xlabel("Channel")
        axs[1, 2].set_ylabel("Time [Blocks]")
        fig.colorbar(im21, ax=axs[1, 2], label="Power [dB]")

        im22 = axs[2, 2].imshow(x_fft_psd_db, **plot_kwargs)
        axs[2, 2].set_title("Standard FFT PSD Output [dB]")
        axs[2, 2].set_xlabel("Channel")
        axs[2, 2].set_ylabel("Time [Blocks]")
        fig.colorbar(im22, ax=axs[2, 2], label="Power [dB]")
        stop()

        plt.suptitle(f"Dirac Comb Leakage Test (Period={delta_period}, Start={delta_start})", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/dirac_comb_leakage_PFB_compare.png", dpi=300, bbox_inches='tight')
        plt.figure(figsize=(10, 6))
        plt.plot(x_PFB_over_psd_db[:, n_chan//2], label='OverPFB PSD (Channel {})'.format(n_chan//2), color='blue', linestyle='--')
        plt.plot(x_PFB_psd_db[:, n_chan//2], label='PFB PSD (Channel {})'.format(n_chan//2), color='green', linestyle='-.')
        plt.scatter((np.arange(x_fft_psd_db.shape[0])), x_fft_psd_db[:, n_chan//2], label='FFT PSD (Channel {})'.format(n_chan//2), color='orange', linewidth=2, alpha=1, marker='x')
        plt.title(f"Temporal Response of Channel N/2 (Period={delta_period}, Start={delta_start})")
        plt.xlabel("Time [Blocks]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # stop()
        plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/dirac_comb_leakage_temporal_compare.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    if waterfall:
        # 5. Plotting the 2x3 Surface (Waterfall) plots
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        plot_kwargs = {
            'aspect': 'auto', 
            'origin': 'lower', 
            'cmap': 'viridis', 
            'interpolation': 'nearest'
        }

        # --- TOP ROW: PFB ---
        
        # Top Left: Raw PFB (Real)
        im00 = axs[0, 0].imshow(X_PFB_real, **plot_kwargs)
        axs[0, 0].set_title("PFB Output (Real Part)")
        axs[0, 0].set_xlabel("Channel")
        axs[0, 0].set_ylabel("Time [Blocks]")
        fig.colorbar(im00, ax=axs[0, 0], label="Amplitude")

        # Top Middle: Raw PFB (Imaginary)
        im01 = axs[0, 1].imshow(X_PFB_imag, **plot_kwargs)
        axs[0, 1].set_title("PFB Output (Imaginary Part)")
        axs[0, 1].set_xlabel("Channel")
        axs[0, 1].set_ylabel("Time [Blocks]")
        fig.colorbar(im01, ax=axs[0, 1], label="Amplitude")

        # Top Right: PFB PSD
        im02 = axs[0, 2].imshow(x_PFB_psd_db, **plot_kwargs)
        axs[0, 2].set_title("PFB PSD Output [dB]")
        axs[0, 2].set_xlabel("Channel")
        axs[0, 2].set_ylabel("Time [Blocks]")
        fig.colorbar(im02, ax=axs[0, 2], label="Power [dB]")

        # --- BOTTOM ROW: Standard FFT ---
        
        # Bottom Left: Raw FFT (Real)
        im10 = axs[1, 0].imshow(x_fft_real, **plot_kwargs)
        axs[1, 0].set_title("Standard FFT Output (Real Part)")
        axs[1, 0].set_xlabel("Channel")
        axs[1, 0].set_ylabel("Time [Blocks]")
        fig.colorbar(im10, ax=axs[1, 0], label="Amplitude")

        # Bottom Middle: Raw FFT (Imaginary)
        im11 = axs[1, 1].imshow(x_fft_imag, **plot_kwargs)
        axs[1, 1].set_title("Standard FFT Output (Imaginary Part)")
        axs[1, 1].set_xlabel("Channel")
        axs[1, 1].set_ylabel("Time [Blocks]")
        fig.colorbar(im11, ax=axs[1, 1], label="Amplitude")

        # Bottom Right: FFT PSD
        im12 = axs[1, 2].imshow(x_fft_psd_db, **plot_kwargs)
        axs[1, 2].set_title("Standard FFT PSD Output [dB]")
        axs[1, 2].set_xlabel("Channel")
        axs[1, 2].set_ylabel("Time [Blocks]")
        fig.colorbar(im12, ax=axs[1, 2], label="Power [dB]")

        plt.suptitle(f"Dirac Comb Leakage Test (Period={delta_period}, Start={delta_start})", fontsize=16)
        plt.tight_layout()
        plt.show()
    if temporal_plot:
        # 6. Plotting the temporal response PSD of a single channel (e.g., Channel N/2 for maximum response)
        plt.figure(figsize=(10, 6))
        plt.plot(x_PFB_psd_db[:, n_chan//2], label='PFB PSD (Channel {})'.format(n_chan//2), color='blue')
        plt.scatter((np.arange(x_fft_psd_db.shape[0])), x_fft_psd_db[:, n_chan//2], label='FFT PSD (Channel {})'.format(n_chan//2), color='orange', linewidth=2, alpha=1, marker='x')
        plt.title(f"Temporal Response of Channel N/2 (Period={delta_period}, Start={delta_start})")
        plt.xlabel("Time [Blocks]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return X_PFB_real, X_PFB_imag, x_PFB_psd_db, x_fft_real, x_fft_imag, x_fft_psd_db

def animate_temporal_dirac_comb(n_taps, n_chan, n_windows, delta_start, N, fps=20, include_noise=False, real=True, is_complex=False, filename="dirac_comb_sweep.mp4"):
    """ 
    Creates an MP4 animation of the 2x3 subplot as the delta_period varies.
    Relies directly on test_temporal_dirac_comb() for data generation.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    plot_kwargs = {
        'aspect': 'auto', 
        'origin': 'lower', 
        'cmap': 'viridis', 
        'interpolation': 'nearest'
    }

    # 1. Initialize the plots with the first frame (delta_period = 256)
    init_period = 256
    
    # Grab data from your test function (setting plotting flags to False)
    X_PFB_real, X_PFB_imag, x_PFB_psd_db, x_fft_real, x_fft_imag, x_fft_psd_db = test_temporal_dirac_comb(
        n_taps, n_chan, n_windows, init_period, delta_start, 
        include_noise=include_noise, real=real, is_complex=is_complex, 
        waterfall=False, temporal_plot=False
    )

    # Top Row: PFB
    im00 = axs[0, 0].imshow(X_PFB_real, **plot_kwargs)
    axs[0, 0].set_title("PFB Output (Real Part)")
    fig.colorbar(im00, ax=axs[0, 0], label="Amplitude")

    im01 = axs[0, 1].imshow(X_PFB_imag, **plot_kwargs)
    axs[0, 1].set_title("PFB Output (Imaginary Part)")
    fig.colorbar(im01, ax=axs[0, 1], label="Amplitude")

    im02 = axs[0, 2].imshow(x_PFB_psd_db, **plot_kwargs)
    axs[0, 2].set_title("PFB PSD Output [dB]")
    fig.colorbar(im02, ax=axs[0, 2], label="Power [dB]")

    # Bottom Row: Standard FFT
    im10 = axs[1, 0].imshow(x_fft_real, **plot_kwargs)
    axs[1, 0].set_title("Standard FFT Output (Real Part)")
    fig.colorbar(im10, ax=axs[1, 0], label="Amplitude")

    im11 = axs[1, 1].imshow(x_fft_imag, **plot_kwargs)
    axs[1, 1].set_title("Standard FFT Output (Imaginary Part)")
    fig.colorbar(im11, ax=axs[1, 1], label="Amplitude")

    im12 = axs[1, 2].imshow(x_fft_psd_db, **plot_kwargs)
    axs[1, 2].set_title("Standard FFT PSD Output [dB]")
    fig.colorbar(im12, ax=axs[1, 2], label="Power [dB]")

    for ax in axs.flat:
        ax.set_xlabel("Channel")
        ax.set_ylabel("Time [Blocks]")

    plt.tight_layout()
    fig.subplots_adjust(top=0.92) 

    # 2. Define the update function for the animation
    def update(frame):
        current_period = 256 + frame
        print(f"Generating frame {frame+1}/{N} with delta_period={current_period}...")
        
        # Call your test function again silently for the new period
        r_pfb, i_pfb, psd_pfb, r_fft, i_fft, psd_fft = test_temporal_dirac_comb(
            n_taps, n_chan, n_windows, current_period, delta_start, 
            include_noise=include_noise, real=real, is_complex=is_complex, 
            waterfall=False, temporal_plot=False
        )

        im00.set_data(r_pfb)
        im01.set_data(i_pfb)
        im02.set_data(psd_pfb)
        im10.set_data(r_fft)
        im11.set_data(i_fft)
        im12.set_data(psd_fft)

        # Dynamically rescale color bars
        im00.set_clim(vmin=np.min(r_pfb), vmax=np.max(r_pfb))
        im01.set_clim(vmin=np.min(i_pfb), vmax=np.max(i_pfb))
        im02.set_clim(vmin=np.min(psd_pfb), vmax=np.max(psd_pfb))
        im10.set_clim(vmin=np.min(r_fft), vmax=np.max(r_fft))
        im11.set_clim(vmin=np.min(i_fft), vmax=np.max(i_fft))
        im12.set_clim(vmin=np.min(psd_fft), vmax=np.max(psd_fft))

        fig.suptitle(f"Dirac Comb Leakage Test (Period={current_period}, Start={delta_start})", fontsize=16)

        return im00, im01, im02, im10, im11, im12

    # 3. Create and save the animation
    filename=f"dirac_comb_sweep_N{N}_fps{fps}.mp4"
    print(f"Generating animation with {N} frames at {fps} fps...")
    ani = animation.FuncAnimation(fig, update, frames=N, blit=False)
    path = "/Users/hilays79/Fourier_Space/Data/"
    ani.save(path+filename, fps=fps, writer='ffmpeg')
    print(f"Animation successfully saved to {filename}")
    plt.close(fig)

def test_spectral_leakage_sine(n_taps, n_chan, n_windows, freq, include_noise=True, plot=True, complex_sine=False):
    """ Test spectral leakage in a PFB spectrometer by generating a sine wave signal and analyzing the resulting spectrum. """
    osr = 32/27 # Oversampling ratio
    data = ts.generate_sine_signal(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, freq=freq, include_noise=include_noise, complex_sine=complex_sine)
    # Apply PFB spectrometer to the generated sine wave signal
    X_psd_PFB = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming", PSD=False)
    X_psd_brute = PFB.brute_force_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming")
    X_psd_fft = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="rectangular", PSD=False)
    x_psd_fft_windowed = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="hamming")
    x_psd_overPFB = OverPFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, osr=osr, n_int=1, window_fn="hamming", PSD=False)
    # x_fine_fft_overPFB = np.fft.fft(x_psd_overPFB[:, 3])
    # plt.plot(np.arange(len(x_fine_fft_overPFB)), np.abs(x_fine_fft_overPFB))
    # plt.yscale("log")
    # plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/temp.png", dpi=300, bbox_inches='tight')
    # stop()
    # plt.plot(np.arange(len(X_psd_PFB[0])), np.abs(X_psd_PFB[0]), label='critical')
    # plt.plot(np.arange(len(x_psd_overPFB[0])), np.abs(x_psd_overPFB[0]), label='over')
    # plt.plot(np.arange(len(X_psd_fft[0])), np.abs(X_psd_fft[0]), label='fft')
    # plt.yscale("log")
    # plt.legend()
    # plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/temp.png", dpi=300, bbox_inches='tight')
    # stop()

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(PFB.db(X_psd_PFB)[0], label='PFB', alpha=1)
        plt.plot(PFB.db(X_psd_brute)[0]-2, label='Brute Force (shifted down by 2 dB for visibility)', alpha=0.5)
        plt.plot(PFB.db(X_psd_fft)[0]+2, label='FFT (shifted up by 2 dB for visibility)', alpha=1)
        plt.plot(OverPFB.db(x_psd_overPFB)[0], label='OverPFB (OSR={:.2f})'.format(osr), alpha=1, linestyle='--')

        # Add some padding to the title to push it up above the new legend space
        plt.title('Spectral Leakage test \n Omega = {}'.format(freq), pad=75) 
        
        plt.ylim(-40, 40)
        plt.xlim(0, n_chan)
        plt.xlabel("Channel")
        plt.ylabel("Power [dB]")
        
        # Position the legend above the plot, centered
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
        
        # Ensure the expanded top area isn't cut off when rendering
        plt.tight_layout() 
        plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/spectral_leakage_sine.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(PFB.db(X_psd_PFB)[0], label='PFB', alpha=1)
        plt.plot(OverPFB.db(x_psd_overPFB)[0], label='OverPFB (OSR={:.2f})'.format(osr), alpha=1, linestyle='--')
        # Add some padding to the title to push it up above the new legend space
        plt.title('Spectral Leakage test \n Omega = {}'.format(freq), pad=75) 
        
        plt.ylim(-40, 40)
        channel_num = freq // (2*np.pi/n_chan)
        plt.xlim(channel_num-2, channel_num+2)
        plt.xlabel("Channel")
        plt.ylabel("Power [dB]")
        
        # Position the legend above the plot, centered
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
        
        # Ensure the expanded top area isn't cut off when rendering
        plt.tight_layout() 
        plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/spectral_leakage_sine_PFB_compare.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    stop()

def test_spectral_leakage_fine_sine(n_taps, n_chan, n_windows, freq, include_noise=True, plot=True, complex_sine=False):
    """ Test spectral leakage in a PFB spectrometer by generating a sine wave signal and analyzing the resulting spectrum. """
    osr = 32/27 # Oversampling ratio
    # Generate the tone
    data = ts.generate_sine_signal(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, freq=freq, include_noise=include_noise, complex_sine=complex_sine)
    # data = ts.generate_dirac_comb_signal(n_taps, n_chan, n_windows, n_taps*n_chan*n_windows, 0, include_noise=include_noise, real=True, is_complex=True)
    # data[0] = data[0] - (1+1j)

    # Apply coarse PFB spectrometer to the generated sine wave signal
    x_pfb_PFB_coarse = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming", PSD=False)
    x_pfb_PFB_coarse_shifted = np.fft.fftshift(x_pfb_PFB_coarse, axes=1)

    # Apply coarse FFT spectrometer to the generated sine wave signal
    x_psd_fft_coarse = PFB.standard_fft_spectrometer(data, n_chan=n_chan, n_int=1, window_fn="rectangular")
    x_psd_coarse_fft_shifted = np.fft.fftshift(x_psd_fft_coarse, axes=1)

    # Apply coarse oversampled PFB spectrometer to the generated sine wave signal
    x_pfb_overPFB_coarse = OverPFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, osr=osr, n_int=1, window_fn="hamming", PSD=False)
    x_pfb_overPFB_coarse_shifted = np.fft.fftshift(x_pfb_overPFB_coarse, axes=1)

    # Apply fine FFT spectrometer to the generated sine wave signal (FFT length equal to input data length for maximum frequency resolution)
    x_psd_fine_fft = PFB.standard_fft_spectrometer(data, n_chan=len(data), n_int=1, window_fn="rectangular")
    x_psd_fine_fft_shifted = np.fft.fftshift(x_psd_fine_fft)

    # Apply fine FFT on coarse data as a second step: PFB
    x_psd_PFB_coarse_shifted_fine_fft = np.abs(np.fft.fft(x_pfb_PFB_coarse_shifted, n=x_pfb_PFB_coarse.shape[0], axis=0))
    x_psd_PFB_coarse_shifted_fine_fft_shifted = np.fft.fftshift(x_psd_PFB_coarse_shifted_fine_fft, axes=0)

    # Apply fine FFT on coarse data as a second step: OverPFB
    x_psd_overPFB_coarse_shifted_fine_fft = np.abs(np.fft.fft(x_pfb_overPFB_coarse_shifted, n=x_pfb_overPFB_coarse_shifted.shape[0], axis=0))
    x_psd_overPFB_coarse_shifted_fine_fft_shifted = np.fft.fftshift(x_psd_overPFB_coarse_shifted_fine_fft, axes=0)

    # Determine fine frequency bins
    freq_fine_fft = np.fft.fftfreq(len(data), d=1/2/np.pi)
    freq_fine_shifted = np.fft.fftshift(freq_fine_fft)

    # Determine coarse frequency bins
    freq_coarse_fft = np.fft.fftfreq(n_chan, d=1/2/np.pi)
    freq_coarse_shifted = np.fft.fftshift(freq_coarse_fft)

    # Determine the coarse fine frequency bins for the second step FFTs
    freq_coarse_fine_PFB_one = np.fft.fftfreq(x_pfb_PFB_coarse_shifted.shape[0], d=n_chan/2/np.pi)
    freq_coarse_fine_PFB_one_shifted = np.fft.fftshift(freq_coarse_fine_PFB_one)
    freq_coarse_fine_OverPFB_one = np.fft.fftfreq(x_pfb_overPFB_coarse_shifted.shape[0], d=n_chan/2/np.pi/osr)
    freq_coarse_fine_OverPFB_one_shifted = np.fft.fftshift(freq_coarse_fine_OverPFB_one)

    # create the final frequency arrays for PFB two-stage cases
    freq_PFB_coarse_shifted_fine_fft_shifted = freq_coarse_fine_PFB_one_shifted[:, np.newaxis] + freq_coarse_shifted[np.newaxis, :]
    freq_overPFB_coarse_shifted_fine_fft_shifted = freq_coarse_fine_OverPFB_one_shifted[:, np.newaxis] + freq_coarse_shifted[np.newaxis, :]

    # arrange in ascending order for plotting
    freq_PFB_coarse_shifted_fine_fft_shifted_sorted_ind = np.argsort(freq_PFB_coarse_shifted_fine_fft_shifted.flatten())
    freq_overPFB_coarse_shifted_fine_fft_shifted_sorted_ind = np.argsort(freq_overPFB_coarse_shifted_fine_fft_shifted.flatten())

    freq_PFB_coarse_shifted_fine_fft_shifted_sorted = freq_PFB_coarse_shifted_fine_fft_shifted.flatten()[freq_PFB_coarse_shifted_fine_fft_shifted_sorted_ind]
    freq_overPFB_coarse_shifted_fine_fft_shifted_sorted = freq_overPFB_coarse_shifted_fine_fft_shifted.flatten()[freq_overPFB_coarse_shifted_fine_fft_shifted_sorted_ind]

    x_PFB_coarse_shifted_fine_fft_shifted_sorted = x_psd_PFB_coarse_shifted_fine_fft_shifted.flatten()[freq_PFB_coarse_shifted_fine_fft_shifted_sorted_ind]
    x_overPFB_coarse_shifted_fine_fft_shifted_sorted = x_psd_overPFB_coarse_shifted_fine_fft_shifted.flatten()[freq_overPFB_coarse_shifted_fine_fft_shifted_sorted_ind] 

    if plot:
        plt.figure(figsize=(10, 6))

        plt.plot(freq_fine_shifted, PFB.db(x_psd_fine_fft_shifted[0]), alpha=1, label='Fine FFT (Length {})'.format(len(data)))
        # plt.plot(freq_coarse_shifted, PFB.db(x_psd_coarse_fft_shifted[0]), label='Coarse FFT (Length {})'.format(n_chan), alpha=1, linestyle='--')
        # plt.plot(freq_coarse_shifted, PFB.db(np.abs(x_pfb_PFB_coarse_shifted[0])), label='PFB (Length {})'.format(n_chan), alpha=1, linestyle='-.')
        # plt.plot(freq_coarse_shifted, PFB.db(np.abs(x_pfb_overPFB_coarse_shifted[0])), label='OverPFB (Length {})'.format(n_chan), alpha=1, linestyle=':')
        plt.plot(freq_PFB_coarse_shifted_fine_fft_shifted_sorted, PFB.db(x_PFB_coarse_shifted_fine_fft_shifted_sorted), label='PFB Coarse + Fine FFT', alpha=1, linestyle='-.')
        plt.plot(freq_overPFB_coarse_shifted_fine_fft_shifted_sorted, PFB.db(x_overPFB_coarse_shifted_fine_fft_shifted_sorted), label='OverPFB Coarse + Fine FFT', alpha=1, linestyle=':')
        plt.vlines(x=freq_coarse_shifted, color='gray', linestyle='--', alpha=0.5, label='Coarse Bin Centers', ymin=-40, ymax=60)
        # plt.ylim(-40, 60)
        plt.xlim(freq-np.pi/n_chan, freq+np.pi/n_chan)
        plt.legend()
        plt.xlabel("Frequency [rad/sample]")
        plt.ylabel("Power [dB]")
        plt.title('Spectral Leakage test \n Omega = {}, osr = {}'.format(freq, osr), pad=75) 
        plt.grid(True, alpha=0.3)
        plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/temp.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return freq_fine_shifted, x_psd_fine_fft_shifted[0], freq_PFB_coarse_shifted_fine_fft_shifted_sorted, x_PFB_coarse_shifted_fine_fft_shifted_sorted, freq_overPFB_coarse_shifted_fine_fft_shifted_sorted, x_overPFB_coarse_shifted_fine_fft_shifted_sorted


def test_channel_response_sweep(n_taps, n_chan, n_windows, freqs, channels_to_plot=(0, 1, 2), include_noise=False, complex_sine=False):
    """
    Sweeps a sine wave across a range of frequencies and records the power
    response in specific PFB channels to visualise the filter shapes.
    """
    # Dictionary to hold the power values for each requested channel
    channel_powers_PFB = {ch: [] for ch in channels_to_plot}
    channel_powers_OverPFB = {ch: [] for ch in channels_to_plot}

    osr = 2 # Oversampling ratio
    
    # Loop over each frequency in the array
    for freq in freqs:
        # 1. Generate the test signal using your framework generator
        data = ts.generate_sine_signal(n_taps=n_taps, 
                                    n_chan=n_chan, 
                                    n_windows=n_windows, 
                                    freq=freq, 
                                    include_noise=include_noise, 
                                    complex_sine=complex_sine)
        
        # 2. Pass the generated data through the PFB spectrometer.
        # Setting n_int = n_windows ensures we get exactly one integrated block back.
        X_psd = PFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, n_int=1, window_fn="hamming")
        X_psd_over = OverPFB.pfb_spectrometer(data, n_taps=n_taps, n_chan=n_chan, osr=osr, n_int=1, window_fn="hamming")
        
        # 3. Extract and store the power for the specific bins we care about
        for ch in channels_to_plot:
            channel_powers_PFB[ch].append(X_psd[0, ch])
            channel_powers_OverPFB[ch].append(X_psd_over[0, ch])

    # 4. Plot the results
    plt.figure(figsize=(10, 6))
    
    for ch in channels_to_plot:
        # Plot the critically sampled PFB and capture the Line2D object
        line = plt.plot(freqs, PFB.db(channel_powers_PFB[ch]), linestyle='-', label=f"Channel {ch} (PFB)")
        
        # Extract the color assigned to that line
        ch_color = line[0].get_color()
        
        # Plot the oversampled PFB using the extracted color and a dashed line
        plt.plot(freqs, OverPFB.db(channel_powers_OverPFB[ch]), linestyle='--', color=ch_color, label=f"Channel {ch} (OverPFB)")

    # Vertical lines at the theoretical center frequencies of the channels
    # bin_center_freq = 2 * pi * k / P
    for ch in channels_to_plot:
        center_freq = 2 * np.pi * ch / n_chan
        if center_freq <= freqs[-1]: # Only plot the line if it falls within our sweep range
            plt.axvline(x=center_freq, color='gray', linestyle='--', alpha=0.5)

    plt.xlim(freqs[0], freqs[-1])
    # Note: What your snippet called 'period' is actually the angular frequency omega (rad/sample)
    plt.xlabel("Frequency $\omega$ [rad/sample]") 
    plt.ylabel("Power [dB]")
    plt.title("PFB Channel Frequency Response Sweep, OSR={:.2f}".format(osr))
    # Position the legend completely outside the plot to the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/channel_response_sweep.png", dpi=300, bbox_inches='tight')
    plt.show()

def get_analytical_channel_response(n_taps, n_chan, channel=1, window_fn="hamming", plot=True):
    """
    Calculates the theoretical, unpadded frequency response of a specific PFB channel 
    directly from the filter tap coefficients.
    """
    M = n_taps
    P = n_chan
    num_taps = M * P

    # Generate the exact prototype filter tap coefficients
    win_coeffs = scipy.signal.get_window(window_fn, num_taps)
    sinc = scipy.signal.firwin(num_taps, cutoff=1.0/P, window="rectangular") 
    prototype_filter = win_coeffs * sinc
    
    # Normalize for processing gain to match empirical spectrometer power levels
    pg = np.sum(np.abs(prototype_filter)**2)
    prototype_filter /= pg**0.5

    n = np.arange(num_taps)

    # Take the standard, unpadded FFT (Length is exactly M * P)
    H_unpadded = np.fft.fft(prototype_filter)

    # Calculate zero-padded FFT for high-resolution smooth curve
    pad_factor = 16 
    N_pad = num_taps * pad_factor
    H_padded = np.fft.fft(prototype_filter, n=N_pad)

    H_db_unpadded = PFB.db(np.real(H_unpadded * np.conj(H_unpadded)))
    H_db_padded = PFB.db(np.real(H_padded * np.conj(H_padded)))

    # 5. Create frequency arrays and shift both arrays so 0 is in the center
    # Unpadded shift
    omega_unpadded = np.fft.fftshift(np.fft.fftfreq(num_taps)) * 2 * np.pi
    H_db_unpadded_shifted = np.fft.fftshift(H_db_unpadded)
    
    # Padded shift
    omega_padded = np.fft.fftshift(np.fft.fftfreq(N_pad)) * 2 * np.pi
    H_db_padded_shifted = np.fft.fftshift(H_db_padded)

    if plot:
        plt.figure(figsize=(10, 6))

        # Plot the smooth zero-padded curve first (so it sits behind the discrete points)
        plt.plot(omega_padded, H_db_padded_shifted, linestyle='-', color='blue', alpha=0.6, label="Theoretical Ch 0 (Zero-Padded)")

        # Plot the discrete unpadded points
        # Changed linestyle to 'none' so the polygonal lines don't clutter the smooth curve, 
        # but you can change it back to '-' if you prefer seeing the connections.
        plt.plot(omega_unpadded, H_db_unpadded_shifted, marker='.', linestyle='none', color='orange', label="Theoretical Ch 0 (Unpadded)")
        
        # Draw a vertical line at the theoretical center frequency for reference
        center_freq = 0
        plt.axvline(x=center_freq, color='gray', linestyle='--', alpha=0.5, label="Center Freq Ch 0")

        # Zoom in to a width of roughly two channels on either side (+/- 2 bins)
        channel_width = 2 * np.pi / P
        plt.xlim(-2 * channel_width, 2 * channel_width) 
        plt.ylim(-40, 40)
        
        plt.xlabel(r"Frequency $\omega$ [rad/sample]")
        plt.ylabel("Power [dB]")
        plt.title("Analytical PFB Channel 0 Response")
        
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/analytical_channel_response.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    return omega_padded, H_db_padded_shifted

if __name__ == "__main__":
    n_taps = 4
    n_chan = 64
    n_windows = 85
    delta_period = n_taps*n_chan*n_windows
    delta_start = n_taps*n_chan*20+1
    # freq=1
    freq = (6)*np.pi/64 # Slightly offset from bin 65's center frequency to demonstrate leakage
    # Angular frequency sweep from 0 to slightly past bin 2's center
    highest_channel = 3
    omegas = np.linspace(0.002, highest_channel*1.1*2*np.pi/n_chan, 1000)
    # test_spectral_leakage_sine(n_taps, n_chan, n_windows, freq, include_noise=False, complex_sine=True)
    # test_spectral_leakage_fine_sine(n_taps, n_chan, n_windows, freq, include_noise=True, complex_sine=True)
    # stop()
    # test_channel_response_sweep(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, freqs=omegas, channels_to_plot=np.arange(0, highest_channel),include_noise=False, complex_sine=True)
    # get_analytical_channel_response(n_taps=n_taps, n_chan=n_chan, channel=0, window_fn="hamming", plot=True)
    test_temporal_dirac_comb(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, delta_period=delta_period, delta_start=delta_start, include_noise=False, real=True, is_complex=False, waterfall=False, temporal_plot=False, overPFB=True)
    # animate_temporal_dirac_comb(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, delta_start=delta_start, N=100, fps=10, include_noise=False, real=True, is_complex=False)




    
    # # ------------ noise test ------------
    # iters = 5000
    # avg_psd_fine_fft = None
    # avg_psd_PFB_fft = None
    # avg_psd_overPFB_fft = None
    # sweep_tone_frqs = np.linspace(0.77, 0.83, iters) # Sweep across the entire frequency range to get an average response across all bins
    # data = np.zeros(n_taps*n_chan*n_windows, dtype=np.complex64) # Pre-allocate data array to avoid overhead in the loop
    # for i in range(iters):
    #     data += ts.generate_sine_signal(n_taps=n_taps, n_chan=n_chan, n_windows=n_windows, freq=sweep_tone_frqs[i], include_noise=True, complex_sine=True)
        
    # freq_fine_fft, psd_fine_fft, freq_PFB_fft, psd_PFB_fft, freq_overPFB_fft, psd_overPFB_fft = test_spectral_leakage_fine_sine(data, n_taps, n_chan, n_windows, sweep_tone_frqs[i], include_noise=False, plot=False, complex_sine=True)

    # plt.figure(figsize=(10, 6))
    # plt.plot(freq_fine_fft, PFB.db(psd_fine_fft), alpha=1, label='Fine FFT (Length {})'.format(len(psd_fine_fft)))
    # plt.plot(freq_PFB_fft, PFB.db(psd_PFB_fft), label='PFB Coarse + Fine FFT', alpha=1)
    # # plt.plot(freq_overPFB_fft, PFB.db(psd_overPFB_fft), label='OverPFB Coarse + Fine FFT', alpha=1)
    # plt.xlim(freq-2*np.pi/n_chan, freq+2*np.pi/n_chan)
    # # plt.ylim(np.max(PFB.db(avg_psd_PFB_fft))-5, np.max(PFB.db(avg_psd_PFB_fft))+5)
    # # plt.ylim([10, 25])
    # plt.xlabel("Frequency [rad/sample]")
    # plt.ylabel("Power [dB]")
    # plt.title('Spectral Leakage test with Noise', pad=75) 
    # plt.grid(True, alpha=0.3)
    # plt.legend(loc='upper right')
    # plt.tight_layout()
    # plt.savefig("/home/hshah/PolyphaseFilterbank/PFB_CPU/codes/PFB_python/images/spectral_leakage_fine_sine_with_sweep.png", dpi=300, bbox_inches='tight')
    # stop()
    