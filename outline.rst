Procedure for PCSVN estimation

For each sample:

    A) Preprocessing
        1) RGB to single channel
            + Luminance transform
            . Isolate green channel (Almoussa, Huynh)
        2) Remove glare 
            a) Mask glare
                + Threshold (*175/255*)
                . Threshold at *80%* of max intensity (Almoussa)
                . Lange (2005) (multistep procedure, done in RGB space actually)
                . Lamprinou (2018) (good overview here)
            b) Post-process mask
                + Dilate with radius *2*
                . Do nothing
            c) Inpaint glare
                + Hybrid inpainting, with size threshold *32*
                . Biharmonic inpainting
                . Mean value of boundary 
                . Median value of boundary 
                . Windowed mean (radius: *15*)
    B) Multiscale Frangi filter
        1) Define parameters
            a) Scales {σ}
                    = n_scales (default: *40*)
                    = scale_range (default *[-2, 3.5]*)
                    = scale_type (*logarithmic base 2* or linear or custom)
                    -> build scales
            b) Betas {β}
                    = *0.5* each scale or custom range
            c) Gammas {γ}
                    = strategy: (half L2 hessian norm or *half hessian frobenius norm*)
                      or custom value each scale
                    = redilate plate per scale (?)
            d) Dilate per scale 
                + Custom function of scale (default *max{10, int(4σ)} if (σ < 20) else int(2σ))*)
                . No dilation
            e) Scale space convolution method
                + Discrete Gaussian kernel with FFT
                . Sampled gaussian kernel with FFT
                . Sample gaussian kernel, standard convolution
        2) For σ in Σ: do Uniscale Frangi Filter
                a) gauss blur image with method from (1e)
                b) take gradient across each axis, take gradient across each axis of gradient to get
                    Hxx, Hxy, Hyy
                c) find eigenvalues of hessian at each point (using np.eig) and sort by magnitude 
                d) zero out principal directions according to Dilate Per Scale
                e) *zero out hessian according to max(ceil(σ),10) INSTEAD* LOOK AT THIS
                f) Calculate Frangi Vesselness Measure
      C) Merging Frangi scores
        1) Calculate Fmax and Fmax.where -> Fmax
        2) Threshold at 95th percentile -> approx
        3) Compare to Trace
        etc.
