// feats.h — Nemo LogMel128 feature extraction (native C++, fftw3f).
// Byte-for-byte matches onnx-asr preprocessors/nemo.py NemoPreprocessor128
// (istupakov/onnx-asr). This preprocessor converts raw waveform -> static
// log-mel features [128, T] which feed the Parakeet-TDT encoder (input
// 'audio_signal' [1,128,T]).
//
// Algorithm (verified against reference):
//   sample_rate 16000, n_fft 512, win_length 400, hop 160, preemph 0.97
//   1. pre-emphasis: x[1:] -= 0.97*x[:-1]
//   2. pad n_fft//2=256 zeros left + right -> length N+512
//   3. STFT (symmetric hann(400) padded 56+56 -> 512), one-sided 257 bins,
//      power = re^2+im^2  -> [T, 257]
//   4. mel: [T,257] @ mel[257,128] (slaney, slaney norm) -> [T,128]
//   5. log(mel + 2^-24) -> [T,128] ; transpose -> [128,T]
//   6. per-mel-band mean/var normalization over TIME; valid frames < lens
//      where lens = N/160 ; scale = 1/(sqrt(var)+1e-5); frames >= lens -> 0
//   T (frame count) = 1 + (N + 512 - 400)/160
#pragma once
#include <fftw3.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace nemo {
static const int SR=16000, NFFT=512, WIN=400, HOP=160, NMELS=128, NBINS=NFFT/2+1; // 257
static const float PREEMPH=0.97f;
static const float LOG_GUARD=1.0f/16777216.0f; // 2^-24

static inline float hz2mel(float f){
    // slaney
    return f<1000.0f ? 3.0f*f/200.0f
                     : 15.0f + 27.0f*std::log(f/1000.0f + 1.19e-7f)/std::log(6.4f);
}
static inline float mel2hz(float m){
    return m<15.0f ? 200.0f*m/3.0f : 1000.0f*std::pow(6.4f,(m-15.0f)/27.0f);
}

// melscale_fbanks(257,0,8000,128,16000,slaney,slaney) -> [257,128]
static inline std::vector<float> mel_fbanks(){
    std::vector<float> fb((size_t)NBINS*NMELS,0.0f);
    // all_freqs = linspace(0,8000,257)
    // m_pts = mel_to_hz( linspace(m_min,m_max,130) )
    float m_min=hz2mel(0.0f), m_max=hz2mel(8000.0f);
    std::vector<float> mpts(NMELS+2), hz(NMELS+2);
    for(int i=0;i<NMELS+2;i++) mpts[i]=m_min+(m_max-m_min)*float(i)/(NMELS+1);
    for(int i=0;i<NMELS+2;i++) hz[i]=mel2hz(mpts[i]);
    for(int b=0;b<NBINS;b++){
        float fbin=8000.0f*float(b)/(NBINS-1); // all_freqs[b]
        for(int m=0;m<NMELS;m++){
            float h=hz[m], c=hz[m+1], t=hz[m+2];
            float up=(fbin-h)/(c-h);
            float dn=(t-fbin)/(t-c);
            float v=std::max(0.0f,std::min(up,dn));
            fb[b*NMELS+m]=v*2.0f/(t-h); // slaney norm
        }
    }
    return fb;
}

// Result: features[128*T] laid out band-major (data[m*T+t]), plus frame
// count T (real STFT frames) and features_len (valid frames = N/160).
// N is the REAL audio sample count; normalization/mask uses `lens` (real), NOT
// a padded length. Callers pad `data` to the static T_frames and feed `lens`
// as the encoder 'length' input so the model masks zero-padded frames.
struct Feats { std::vector<float> data; long T; long lens; };

// Pad/retain feature rows (band-major, each row of length T) so the time axis
// has exactly `T_target` frames (zero-fill the pad on the right). Returns the
// real lens unchanged.
static inline void pad_to(Feats& f, long T_target){
    if(f.T>=T_target) return;
    std::vector<float> nu((size_t)NMELS*T_target,0.0f);
    for(int m=0;m<NMELS;m++)
        for(long t=0;t<f.T;t++) nu[(size_t)m*T_target+t]=f.data[(size_t)m*f.T+t];
    f.data.swap(nu); f.T=T_target;
}

// N = number of SAMPLES in the REAL (unpadded) waveform.
static inline Feats extract(const std::vector<float>& wav, long N){
    // 1. pre-emphasis (over the real N samples)
    std::vector<float> x(N);
    if(N>0) x[0]=wav[0];
    for(long i=1;i<N;i++) x[i]=wav[i]-PREEMPH*wav[i-1];

    // 2. pad 256 left + right -> length N+512
    std::vector<float> xp(N+NFFT,0.0f);
    for(long i=0;i<N;i++) xp[NFFT/2+i]=x[i];

    // frame count T = 1 + (N+512-400)/160
    long T = std::max<long>(1, 1 + (N+NFFT-WIN)/HOP);
    long lens = N/HOP; // features_lens (valid frames)

    // hann(400) symmetric, padded 56+56 -> 512
    std::vector<float> w(NFFT,0.0f);
    for(int i=0;i<WIN;i++) w[(NFFT-WIN)/2+i] = 0.5f*(1.0f-std::cos(2.0f*M_PI*i/(WIN-1)));

    fftwf_complex* in=(fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*NFFT);
    fftwf_complex* out=(fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*NFFT);
    fftwf_plan plan=fftwf_plan_dft_1d(NFFT,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
    auto mel=mel_fbanks();

    // power spectrogram [T][257]
    std::vector<float> spec((size_t)T*NBINS,0.0f);
    for(long t=0;t<T;t++){
        long off=t*HOP;
        for(int n=0;n<NFFT;n++){
            double v = (off+n<(long)xp.size()) ? double(xp[off+n])*w[n] : 0.0;
            in[n][0]=(float)v; in[n][1]=0.0f;
        }
        fftwf_execute(plan);
        for(int b=0;b<NBINS;b++){ float re=out[b][0],im=out[b][1]; spec[t*NBINS+b]=re*re+im*im; }
    }
    fftwf_destroy_plan(plan); fftwf_free(in); fftwf_free(out);

    // mel -> log -> transpose to [128,T]
    Feats f; f.T=T; f.lens=lens;
    f.data.assign((size_t)NMELS*T,0.0f);
    for(long t=0;t<T;t++)
        for(int m=0;m<NMELS;m++){
            float acc=0;
            for(int b=0;b<NBINS;b++) acc += spec[t*NBINS+b]*mel[b*NMELS+m];
            f.data[(size_t)m*T + t] = std::log(acc + LOG_GUARD);
        }

    // normalize over time per band, valid frames [0,lens)
    for(int m=0;m<NMELS;m++){
        double mean=0; for(long t=0;t<lens;t++) mean+=f.data[(size_t)m*T+t];
        mean/= (lens>0?lens:1);
        double var=0; for(long t=0;t<lens;t++){ double d=f.data[(size_t)m*T+t]-mean; var+=d*d; }
        var/= (lens>1?lens-1:1);
        double s=1.0/(std::sqrt(var)+1e-5);
        for(long t=0;t<lens;t++) f.data[(size_t)m*T+t]=(float)((f.data[(size_t)m*T+t]-mean)*s);
        for(long t=lens;t<T;t++) f.data[(size_t)m*T+t]=0.0f; // mask frames >= lens
    }
    return f;
}
} // namespace nemo
