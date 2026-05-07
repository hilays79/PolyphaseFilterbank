// Wrapper functions for FFTW library to perform FFT operations based on output precision.
#pragma once
#include <fftw3.h>

// --- FFTW WRAPPER (Handles 64-bit and 32-bit switching) ---
template <typename T> struct FFTWWrapper; // General template for compilation, never used during execution

// Specialization for 64-bit double, <> means it is a specialzation and <double> means this specialization is for double precision
template <> struct FFTWWrapper<double> { 
    using complex_type = fftw_complex; // This line is there to easily define the type of pointer in PFB.cpp, using is essentially nicknaming the type fftw_complex to complex_type.    
    static inline fftw_plan plan_many_dft(int rank, const int *n, int howmany,
                                   complex_type *in, const int *inembed, int istride, int idist,
                                   complex_type *out, const int *onembed, int ostride, int odist,
                                   int sign, unsigned flags) {
        return fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
    }
    
    static inline void execute(fftw_plan p) { fftw_execute(p); } // Note that this is a function call declaration and definition in one. Void needed for declaration.
    static inline void destroy_plan(fftw_plan p) { fftw_destroy_plan(p); }
};

// Specialization for 32-bit float
template <> struct FFTWWrapper<float> {
    using complex_type = fftwf_complex;
    
    static inline fftwf_plan plan_many_dft(int rank, const int *n, int howmany,
                                   complex_type *in, const int *inembed, int istride, int idist,
                                   complex_type *out, const int *onembed, int ostride, int odist,
                                   int sign, unsigned flags) {
        return fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags);
    }
    
    static inline void execute(fftwf_plan p) { fftwf_execute(p); }
    static inline void destroy_plan(fftwf_plan p) { fftwf_destroy_plan(p); }
};