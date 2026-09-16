// dictate.cpp — Native (C++/OpenVINO) Parakeet-TDT dictation runtime.
//
// Designed after Handy's architecture:
//   - App stays alive (lightweight process monitoring the Copilot hotkey)
//   - Model loads ONLY when you press the hotkey (load-on-demand)
//   - Model unloads immediately after transcription (no warm daemon)
//   - Menu bar mic icon: only visible while recording / transcribing
//   - Toggle flow: 1st tap = start recording, 2nd tap = stop + transcribe
//
// Pipeline:
//   raw audio -> [VAD prefilter: Silero VAD, frame-by-frame] -> speech-only audio
//   -> feats.h logmel [128,T] -> encoder (NPU) -> [1,T_enc,1024]
//   -> TDT greedy decode (decoder_joint, CPU) -> tokens -> text
//
// Silero VAD parameters (matching Handy):
//   - VAD threshold: 0.3 (probability of speech)
//   - VAD frame: 512 samples (~32ms at 16kHz)
//   - Pre-roll/prefill: 450ms before speech onset
//   - Hangover: 450ms after speech ends (captures trailing sounds)
//   - Onset: 1 voiced frame needed to trigger speech start
//
// Usage:
//   dictate --daemon                 stay alive, listen for Copilot toggle
//   dictate --record <file.wav>      record from mic until Ctrl-C, save wav
//   dictate --transcribe <file.wav>  print transcript
//   dictate --type <file.wav>        transcribe + type + clipboard
//   dictate --device NPU|CPU        select encoder device
//   dictate --max-secs N             recording window (default 30s)
//   dictate --vad-threshold F        VAD speech probability threshold (default 0.3)
//   dictate --vad-off                disable VAD prefiltering
#include <openvino/openvino.hpp>
#include <openvino/core/partial_shape.hpp>
#include "feats.h"
#include <sndfile.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <deque>
#include <unistd.h>
#include <signal.h>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/select.h>

static std::string MODEL_DIR = "/home/shlok/.local/share/npu-asr/models/parakeet-tdt-0.6b-v3-onnx";
static std::string VAD_DIR     = "/home/shlok/.local/share/npu-asr/models/vad";
static std::string CACHE_DIR   = "/home/shlok/.local/share/npu-asr/cache";
static std::string DEVICE      = "NPU";
static double MAX_SECS         = 30.0;  // recording window (service passes --max-secs)
// (service passes 600s via --max-secs)

static long   N_MAX            = 480000; // MAX_SECS*16000
static long   T_MAX            = 3001;  // feats frames for N_MAX
static bool   VAD_ENABLED      = true;  // VAD prefiltering
static float  VAD_THRESHOLD    = 0.3f;  // speech probability threshold (matches Handy)

static const int VOCAB = 8193, BLANK = 8192, MAX_TOKENS_PER_STEP = 10;
static const int STATE_DIM = 640;

// Silero VAD parameters (matching Handy)
static const size_t VAD_FRAME_SAMPLES = 512;  // ~32ms at 16kHz
static const int    VAD_SR = 16000;
static const int    VAD_PREFILL_MS = 450;     // pre-roll buffer before speech
static const int    VAD_HANGOVER_MS = 450;    // post-speech tail
static const int    VAD_ONSET_FRAMES = 2;     // voiced frames to confirm onset

// ---------------- vocab ----------------
struct Vocab {
    std::vector<std::string> tok;
    int size=0, blank=BLANK;
    bool load(const std::string& path);
};
bool Vocab::load(const std::string& path){
    std::ifstream f(path);
    if(!f) return false;
    std::string line;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        auto sp=line.rfind(' ');
        std::string s = (sp==std::string::npos)? line : line.substr(0,sp);
        // decode: U+2581 -> space
        std::string out; out.reserve(s.size());
        size_t i=0;
        while(i<s.size()){
            unsigned char c=(unsigned char)s[i];
            if(c==0xE2 && i+2<s.size() && (unsigned char)s[i+1]==0x96 && (unsigned char)s[i+2]==0x81){
                out.push_back(' '); i+=3; continue;
            }
            size_t n=(c<0x80)?1:((c<0xE0)?2:((c<0xF0)?3:4));
            for(size_t k=0;k<n && i<s.size();k++) out.push_back(s[i++]);
        }
        tok.push_back(out);
    }
    size=tok.size(); return size>0;
}

// text post-processing (matches _decode_tokens)
static std::string ids_to_text(const Vocab& v, const std::vector<int>& ids){
    std::string s;
    for(int i: ids) s += v.tok[i];
    std::string out; out.reserve(s.size());
    for(size_t i=0;i<s.size();i++){
        char c=s[i];
        if(c==' '){
            if(i>0 && i+1<s.size()){
                char nx=s[i+1];
                bool word = (nx>='a'&&nx<='z')||(nx>='A'&&nx<='Z')||(nx>='0'&&nx<='9')||nx=='_';
                if(word) out.push_back(' ');
            }
            continue;
        }
        out.push_back(c);
    }
    return out;
}

// ---------------- Silero VAD (OpenVINO) ----------------
// Wraps the silero_vad_v4.onnx model with stateful LSTM tracking.
// Uses 512-sample (32ms @ 16kHz) frames, matching Handy's SileroVad backend.
struct SileroVad {
    ov::Core core;
    std::unique_ptr<ov::CompiledModel> compiled;
    std::unique_ptr<ov::InferRequest> request;
    std::vector<float> h_state;  // LSTM hidden state [2,1,64]
    std::vector<float> c_state;  // LSTM cell state [2,1,64]
    float threshold;

    // Smoothing state (mirrors Handy's SmoothedVad)
    std::deque<std::vector<float>> prefill_buffer;  // buffered frames for pre-roll
    size_t hangover_remaining;   // remaining hangover frames after speech ends
    size_t onset_counter;        // consecutive voiced frames for onset detection
    bool in_speech;              // currently in a speech segment
    size_t prefill_frames_actual;  // computed prefill frame count

    bool load(const std::string& model_path, float vad_threshold);
    bool is_speech(const float* samples, size_t count);  // returns true if speech detected
    void reset();  // clear all state
};

bool SileroVad::load(const std::string& model_path, float vad_threshold){
    threshold = vad_threshold;

    // Initialize LSTM states to zero
    h_state.assign(2 * 1 * 64, 0.0f);
    c_state.assign(2 * 1 * 64, 0.0f);

    // Load and reshape the model for 512-sample frames
    auto model = core.read_model(model_path);
    std::map<std::string, ov::PartialShape> shapes;
    shapes["input"] = ov::PartialShape({1, VAD_FRAME_SAMPLES});
    shapes["sr"] = ov::PartialShape({});
    shapes["h"] = ov::PartialShape({2, 1, 64});
    shapes["c"] = ov::PartialShape({2, 1, 64});
    model->reshape(shapes);

    try {
        compiled = std::make_unique<ov::CompiledModel>(core.compile_model(model, "CPU"));
        request = std::make_unique<ov::InferRequest>(compiled->create_infer_request());

        // Set sample rate once (it's static)
        request->get_tensor("sr").data<int64_t>()[0] = VAD_SR;

        fprintf(stderr, "[vad] Silero VAD loaded (threshold=%.2f, frame=%zums)\n",
                threshold, VAD_FRAME_SAMPLES * 1000 / VAD_SR);
    } catch (const std::exception& e) {
        fprintf(stderr, "[vad] Failed to load Silero VAD: %s\n", e.what());
        return false;
    }

    // Compute prefill frame count for smoothing
    size_t prefill_frames = (VAD_PREFILL_MS * VAD_SR) / (1000 * VAD_FRAME_SAMPLES);
    if (prefill_frames < 1) prefill_frames = 1;
    prefill_frames_actual = prefill_frames;
    prefill_buffer.clear();
    prefill_buffer.resize(0);  // deque, no reserve

    size_t hangover_frames = (VAD_HANGOVER_MS * VAD_SR) / (1000 * VAD_FRAME_SAMPLES);
    if (hangover_frames < 1) hangover_frames = 1;

    hangover_remaining = 0;
    onset_counter = 0;
    in_speech = false;

    fprintf(stderr, "[vad] prefill_frames=%zu hangover_frames=%zu onset_frames=%d\n",
            prefill_frames, hangover_frames, VAD_ONSET_FRAMES);

    return true;
}

bool SileroVad::is_speech(const float* samples, size_t count){
    if (count < VAD_FRAME_SAMPLES) return false;

    // Copy audio to input tensor
    auto input_tensor = request->get_tensor("input");
    float* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < VAD_FRAME_SAMPLES; i++) {
        input_data[i] = samples[i];
    }

    // Copy LSTM states
    auto h_tensor = request->get_tensor("h");
    auto c_tensor = request->get_tensor("c");
    std::copy(h_state.begin(), h_state.end(), h_tensor.data<float>());
    std::copy(c_state.begin(), c_state.end(), c_tensor.data<float>());

    // Run inference
    request->infer();

    // Read output probability
    float prob = request->get_tensor("output").data<float>()[0];

    // Read and store new LSTM states
    auto hn_tensor = request->get_tensor("hn");
    auto cn_tensor = request->get_tensor("cn");
    std::copy(hn_tensor.data<float>(), hn_tensor.data<float>() + hn_tensor.get_size(), h_state.begin());
    std::copy(cn_tensor.data<float>(), cn_tensor.data<float>() + cn_tensor.get_size(), c_state.begin());

    // Determine if this frame has speech
    bool frame_has_speech = prob > threshold;

    // --- Smoothing logic (mirrors Handy's SmoothedVad) ---
    // Buffer this frame for potential pre-roll emission
    std::vector<float> frame_buf(samples, samples + VAD_FRAME_SAMPLES);
    prefill_buffer.push_back(std::move(frame_buf));

    // Keep only the last prefill_frames+1 frames
    while (prefill_buffer.size() > prefill_frames_actual + 1) {
        prefill_buffer.pop_front();
    }

    if (in_speech) {
        if (frame_has_speech) {
            // Ongoing speech - reset hangover
            hangover_remaining = (VAD_HANGOVER_MS * VAD_SR) / (1000 * VAD_FRAME_SAMPLES);
        } else if (hangover_remaining > 0) {
            // In hangover tail - still speech
            hangover_remaining--;
        } else {
            // Speech ended
            in_speech = false;
        }
        return true;
    } else {
        if (frame_has_speech) {
            onset_counter++;
            if (onset_counter >= VAD_ONSET_FRAMES) {
                // Confirmed speech onset - emit prefill frames + current
                in_speech = true;
                hangover_remaining = (VAD_HANGOVER_MS * VAD_SR) / (1000 * VAD_FRAME_SAMPLES);
                onset_counter = 0;
                return true;
            }
            // Accumulating voiced frames for onset - not yet speech
            return false;
        } else {
            onset_counter = 0;
            return false;
        }
    }
}

void SileroVad::reset(){
    std::fill(h_state.begin(), h_state.end(), 0.0f);
    std::fill(c_state.begin(), c_state.end(), 0.0f);
    prefill_buffer.clear();
    prefill_frames_actual = 0;
    hangover_remaining = 0;
    onset_counter = 0;
    in_speech = false;
}

// ---------------- audio read ----------------
static std::vector<float> read_audio(const std::string& path, long* out_rate){
    SF_INFO info; memset(&info,0,sizeof(info));
    SNDFILE* sf=sf_open(path.c_str(),SFM_READ,&info);
    if(!sf) throw std::runtime_error("sf_open failed: "+path);
    std::vector<float> tmp((size_t)info.frames*info.channels);
    sf_readf_float(sf,tmp.data(),info.frames);
    sf_close(sf);
    long rate=info.samplerate;
    std::vector<float> mono(info.frames);
    for(long i=0;i<info.frames;i++){
        float s=0; for(int c=0;c<info.channels;c++) s+=tmp[i*info.channels+c];
        mono[i]=s/info.channels;
    }
    if(rate==16000){ *out_rate=rate; return mono; }
    double ratio=double(rate)/16000.0;
    long n=(long)std::floor(mono.size()/ratio);
    std::vector<float> out; out.reserve(n);
    for(long i=0;i<n;i++){
        double pos=i*ratio; long i0=(long)pos,i1=std::min<long>(i0+1,mono.size()-1);
        double fr=pos-i0; out.push_back((float)(mono[i0]*(1-fr)+mono[i1]*fr));
    }
    *out_rate=16000; return out;
}

// ---------------- ASR engine (load-on-demand) ----------------
struct Asr {
    ov::Core core;
    std::unique_ptr<ov::CompiledModel> enc_cm;
    std::unique_ptr<ov::CompiledModel> dec_cm;
    std::unique_ptr<ov::InferRequest> enc_req;
    std::unique_ptr<ov::InferRequest> dec_req;
    Vocab vocab;
    long T_frames;
    std::vector<float> st1, st2;  // decoder LSTM state
    std::string DEVICE;

    ~Asr() { unload_models(); }

    void load_models(const std::string& device, long t_frames);
    void unload_models();
    std::string transcribe(const std::vector<float>& feats, long T, long lens);
};

void Asr::load_models(const std::string& device, long t_frames){
    if(!vocab.tok.empty()) return; // already loaded
    T_frames=t_frames;
    DEVICE = device;
    core.set_property(ov::cache_dir(CACHE_DIR));
    vocab.load(MODEL_DIR+"/vocab.txt");
    fprintf(stderr,"[asr] vocab=%d blank=%d\n", vocab.size, vocab.blank);

    // encoder
    auto enc=core.read_model(MODEL_DIR+"/encoder-model.int8.onnx");
    std::map<std::string,ov::PartialShape> np;
    np["audio_signal"]=ov::PartialShape{1, nemo::NMELS, T_frames};
    np["length"]=ov::PartialShape{1};
    enc->reshape(np);
    enc_cm=std::make_unique<ov::CompiledModel>(core.compile_model(enc, device));
    enc_req=std::make_unique<ov::InferRequest>(enc_cm->create_infer_request());
    fprintf(stderr,"[asr] encoder compiled on %s\n", device.c_str());

    // decoder always on CPU (NPU INT8 LSTM has quantization accuracy issues)
    auto dec=core.read_model(MODEL_DIR+"/decoder_joint-model.int8.onnx");
    std::map<std::string,ov::PartialShape> nd;
    for(auto&i:dec->inputs()){
        auto n=i.get_any_name();
        if(n=="encoder_outputs") nd[n]=ov::PartialShape{1,1024,1};
        else if(n=="targets")    nd[n]=ov::PartialShape{1,1};
        else if(n=="target_length") nd[n]=ov::PartialShape{1};
        else if(n=="input_states_1"||n=="input_states_2") nd[n]=ov::PartialShape{2,1,STATE_DIM};
    }
    dec->reshape(nd);
    dec_cm=std::make_unique<ov::CompiledModel>(core.compile_model(dec, "CPU"));
    dec_req=std::make_unique<ov::InferRequest>(dec_cm->create_infer_request());
    fprintf(stderr,"[asr] decoder compiled on CPU\n");
}

void Asr::unload_models(){
    if(dec_req) dec_req.reset();
    if(dec_cm) dec_cm.reset();
    if(enc_req) enc_req.reset();
    if(enc_cm) enc_cm.reset();
    st1.clear(); st2.clear();
    fprintf(stderr,"[asr] models unloaded\n");
}

std::string Asr::transcribe(const std::vector<float>& feats, long T, long lens){
    /*
     * feed encoder (shapes are static after compile_model; just memcpy data)
     */
    memcpy(enc_req->get_tensor("audio_signal").data<float>(), feats.data(),
           sizeof(float)*feats.size());
    enc_req->get_tensor("length").data<int64_t>()[0]=lens;
    enc_req->infer();
    auto enc_out=enc_req->get_tensor("outputs");
    auto enc_len_t=enc_req->get_tensor("encoded_lengths");
    auto esh=enc_out.get_shape();
    long E=esh[2];
    long enc_len=enc_len_t.data<int64_t>()[0];
    if(enc_len>E) enc_len=E;
    std::vector<float> encodings((size_t)E*1024);
    {
        const float* src=enc_out.data<float>();
        for(long e=0;e<E;e++) for(int d=0;d<1024;d++)
            encodings[e*1024+d]=src[d*E+e];
    }

    /*
     * TDT greedy decode
     */
    st1.assign(2*1*640,0.0f); st2.assign(2*1*640,0.0f);
    std::vector<int> tokens;
    long t=0; int emitted=0;

    while(t<enc_len){
        // decoder_joint (shapes are static; just memcpy data)
        {
            auto te=dec_req->get_tensor("encoder_outputs");
            memcpy(te.data<float>(), &encodings[t*1024], sizeof(float)*1024);

            int prev = tokens.empty()? vocab.blank : tokens.back();
            auto tt=dec_req->get_tensor("targets");
            int32_t* tp=tt.data<int32_t>(); tp[0]=prev;

            auto tl=dec_req->get_tensor("target_length");
            tl.data<int32_t>()[0]=1;

            auto s1=dec_req->get_tensor("input_states_1");
            memcpy(s1.data<float>(), st1.data(), sizeof(float)*st1.size());
            auto s2=dec_req->get_tensor("input_states_2");
            memcpy(s2.data<float>(), st2.data(), sizeof(float)*st2.size());
        }
        dec_req->infer();

        auto o_t=dec_req->get_tensor("outputs");
        auto osh=o_t.get_shape();
        size_t outer=ov::shape_size(osh)/8198;
        const float* o=o_t.data<float>();
        const float* row=o + (outer-1)*8198;

        int token=-1; float token_max=-1e30f;
        for(int i=0;i<BLANK;i++){ if(row[i]>token_max){ token_max=row[i]; token=i; } }
        int dur_idx=0; float dur_max=-1e30f;
        for(int i=VOCAB;i<8198;i++){ if(row[i]>dur_max){ dur_max=row[i]; dur_idx=i-VOCAB; } }
        int step=dur_idx;

        if(token!=vocab.blank){
            auto os1=dec_req->get_tensor("output_states_1");
            auto os2=dec_req->get_tensor("output_states_2");
            st1.assign(os1.data<float>(), os1.data<float>()+ov::shape_size(os1.get_shape()));
            st2.assign(os2.data<float>(), os2.data<float>()+ov::shape_size(os2.get_shape()));
            tokens.push_back(token);
            emitted++;
        }
        if(step>0){ t+=step; emitted=0; }
        else if(token==vocab.blank || emitted==MAX_TOKENS_PER_STEP){ t+=1; emitted=0; }
    }
    return ids_to_text(vocab, tokens);
}

// ---------------- clipboard / typing ----------------

// ─── Model helpers for parakeet-stream backend ───
static std::string MODELS_DIR = "/home/shlok/.local/share/npu-asr/models";

static std::string read_state_str(const std::string& name, const std::string& fallback=""){
    std::string p = "/home/shlok/.local/share/npu-asr/state/" + name;
    std::ifstream f(p);
    if(!f) return fallback;
    std::string s;
    std::getline(f, s);
    while(!s.empty() && (s.back()==' '||s.back()=='\n'||s.back()=='\r'||s.back()=='\t')) s.pop_back();
    while(!s.empty() && (s.front()==' '||s.front()=='\t')) s.erase(s.begin());
    return s;
}

static std::string read_model(){
    return read_state_str("model.txt", "whisper-base-en");
}

static std::string read_device(){
    std::string s = read_state_str("device.txt", DEVICE);
    if(s.empty()) return DEVICE;
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if(upper != "CPU" && upper != "GPU" && upper != "NPU") return DEVICE;
    return upper;
}

static bool read_enabled(){
    std::string s = read_state_str("enabled", "on");
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return !(s == "off" || s == "false" || s == "0" || s == "disabled");
}

// A model that is NOT explicitly English-only is multilingual (tiny, turbo, qwen).
static bool model_is_multilingual(const std::string& dir){
    std::string gc = dir + "/generation_config.json";
    FILE* p = fopen(gc.c_str(), "r");
    if(!p) return false;
    std::string s; char buf[5120]; size_t n;
    while((n = fread(buf,1,sizeof(buf),p)) > 0) s.append(buf,n);
    fclose(p);
    if(s.find("\"is_multilingual\":false") != std::string::npos ||
       s.find("\"is_multilingual\": false") != std::string::npos)
        return false;
    return true;
}

static std::string read_language(){
    return read_state_str("language.txt", "");
}

static std::string resolve_model_dir(const std::string& model_name){
    if(model_name.empty()) return "";
    std::string dir = MODELS_DIR + "/" + model_name;
    if(::access((dir + "/devices.txt").c_str(), R_OK) == 0) return dir;
    if(::access(dir.c_str(), R_OK | X_OK) == 0) return dir;
    return "";
}

static std::string model_backend(const std::string& model_name){
    if(model_name.find("whisper") == 0) return "whisper";
    if(model_name.find("qwen3-asr") == 0) return "qwen3";
    if(model_name == "parakeet-v3") return "parakeet-stream";
    if(model_name.find("parakeet") == 0) return "parakeet-ov";
    if(model_name == "nepali-indicwav2vec") return "indicwav2vec";
    return "unknown";
}

// Detect if focused window is a terminal (needs Ctrl+Shift+V instead of Ctrl+V)
static bool focused_is_terminal(){
    FILE* p = popen("hyprctl activewindow -j", "r");
    if(!p) return false;
    char buf[4096]; size_t n = fread(buf,1,sizeof(buf)-1,p); buf[n]=0; pclose(p);
    std::string s(buf);
    static const char* terms[] = {"foot","kitty","alacritty","wezterm","konsole","gnome-terminal"};
    for(auto t: terms) if(s.find(t)!=std::string::npos) return true;
    return false;
}

// Paste clipboard contents — terminal needs Ctrl+Shift+V, others need Ctrl+V
static void paste_clipboard(){
    if(focused_is_terminal())
        system("wtype -M ctrl -M shift -k v -m shift -m ctrl");
    else
        system("wtype -M ctrl -k v -m ctrl");
}

// ─── audio feedback ─────────────────────────────
// Fire-and-forget sound playback: double-fork + exec so the player
// never delays transcription or blocks on audio I/O. Silently skips
// if the sound file or a player binary is missing — dictation never
// fails or stalls because of audio feedback.
//
// Pop start/stop pair from Handy (pop_start.wav/pop_stop.wav), copied
// to the sounds dir. Start plays on mic open; stop plays at the
// exact record-stop trigger moment (before the 1s audio tail), so the
// user hears the blip immediately on tap — the async playback overlaps
// the tail with no timing impact.
//
// Volume: read from state/volume at each playback (plain integer 0-100,
// default 15). Linear mapping: gain = slider/100. The user found all perceptual
// curves (quadratic, power, piecewise) too steep — the bottom two-thirds of
// the slider collapsed to near-silence. A linear gain gives a uniform feel:
// 100% = 1.0 (0dB, full), 50% = 0.5 (-6dB), 25% = 0.25 (-12dB, clearly
// audible), 15% = 0.15 (-16dB, soft but present), 0% = 0 (silent).
// Applied via player-native flags:
//   paplay   --volume=<0..65536>  (gain * 65536)
//   pw-play  --volume=<0.0..1.0>  (gain)
//   ffplay   -volume <0..100>     (gain * 100)
//   aplay    — no volume flag; falls back to file amplitude (rare, last resort)
// ──────────────────────────────────────────────────────────────

// Read audio-feedback volume from state file. Missing/unparseable → 15 (default).
static int read_volume(){
    static const char* VOL_FILE = "/home/shlok/.local/share/npu-asr/state/volume";
    FILE* f = fopen(VOL_FILE, "r");
    if(!f) return 15;
    int vol = 15;
    if(fscanf(f, "%d", &vol) != 1) vol = 15;
    fclose(f);
    if(vol < 0) vol = 0;
    if(vol > 100) vol = 100;
    return vol;
}

// Piecewise linear gain from slider (0-100) to amplitude gain.
// Target aural points (interpolated linearly between knots):
//   100% -> 1.0000, 50% -> 0.6000, 10% -> 0.4000, 1% -> 0.1000, 0% -> 0.0000
// Previous curves (linear, piecewise 100/80/60/40/0) were rejected —
// the bottom half collapsed to near-silence. This curve gives the user
// a perceptually even spread across the whole slider range.
static double volume_to_gain(double slider){
    if(slider <= 0.0)    return 0.0000;  // 0% -> 0.0
    if(slider >= 100.0)  return 1.0000;  // 100% -> 1.0
    if(slider >= 50.0)   return 0.6000 + (slider - 50.0)  * (1.0000 - 0.6000) / (100.0 - 50.0);  // 50%->100%
    if(slider >= 10.0)   return 0.4000 + (slider - 10.0)  * (0.6000 - 0.4000) / (50.0 - 10.0);   // 10%->50%
    if(slider >= 1.0)    return 0.1000 + (slider - 1.0)   * (0.4000 - 0.1000) / (10.0 - 1.0);    // 1%->10%
    return 0.1000 * slider / 1.0;                                                              // 0%->1%:  0.0->0.1
}

static void play_sound_async(const char* label, const char* which){
    static const char* SOUND_BASE = "/home/shlok/.local/share/npu-asr/sounds";
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.wav", SOUND_BASE, which);
    if(access(path, R_OK) != 0){
        fprintf(stderr, "[asr-audio] %s: sound file %s not found, skipping\n", label, path);
        fflush(stderr);
        return;
    }
    int slider = read_volume();
    double gain = volume_to_gain((double)slider);
    fprintf(stderr, "[asr-audio] %s: playing %s slider=%d%% gain=%.4f (%.1f dB)\n",
            label, path, slider, gain, 20.0 * log10(gain));
    fflush(stderr);

    // Format player-native volume args (declared before fork so they survive).
    // paplay: 0..65536, pw-play: 0.0..1.0, ffplay: 0..100
    char pa_vol[32]; snprintf(pa_vol, sizeof(pa_vol), "--volume=%d", (int)(gain * 65536));
    char pw_vol[32]; snprintf(pw_vol, sizeof(pw_vol), "--volume=%.4f", gain);
    char ff_vol[16]; snprintf(ff_vol, sizeof(ff_vol), "%d", (int)(gain * 100));

    // Block SIGCHLD around the fork so the daemon's SIGCHLD handler
    // doesn't set g_rec_end from our intermediate child's exit.
    sigset_t set, oldset;
    sigemptyset(&set);
    sigaddset(&set, SIGCHLD);
    sigprocmask(SIG_BLOCK, &set, &oldset);
    pid_t mid = fork();
    if(mid < 0){
        sigprocmask(SIG_SETMASK, &oldset, NULL);
        return; // fork failed — silently skip (no audio, no delay)
    }
    if(mid > 0){
        // Parent: reap the middle child so it doesn't zombie, then restore.
        int status;
        waitpid(mid, &status, WNOHANG);
        sigprocmask(SIG_SETMASK, &oldset, NULL);
        return;
    }
    // Middle child: restore SIGCHLD, setsid, fork again, exec player.
    sigprocmask(SIG_SETMASK, &oldset, NULL);
    setsid();
    pid_t player = fork();
    if(player < 0) _exit(0);
    if(player > 0) _exit(0); // middle child exits immediately
    // Grandchild: try players in order of preference, passing native volume.
    execlp("paplay", "paplay", pa_vol, path, (char*)NULL);
    execlp("pw-play", "pw-play", pw_vol, path, (char*)NULL);
    execlp("aplay", "aplay", path, (char*)NULL);
    execlp("ffplay", "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", ff_vol, path, (char*)NULL);
    _exit(127);
}
static void play_start_sound(){ play_sound_async("start", "start"); }
static void play_stop_sound(){ play_sound_async("stop", "stop"); }

static std::string transcribe_via_helper(const std::string& helper,
    const std::string& model_dir, const std::string& device,
    const std::string& wav, const std::string& lang){
    std::string cmd = "python3 " + helper + " --model-dir " + model_dir +
        " --device " + device + " --audio " + wav;
    if(!lang.empty()) cmd += " --language " + lang;
    cmd += " 2>/dev/null";

    FILE* p = popen(cmd.c_str(), "r");
    if(!p) {
        fprintf(stderr, "[asr-helper] popen failed: %s\n", strerror(errno));
        return "";
    }

    std::string output;
    char buf[4096];
    while(fgets(buf, sizeof(buf), p)) output += buf;
    int rc = pclose(p);

    if(rc != 0) {
        fprintf(stderr, "[asr-helper] helper exited %d\n", rc);
        return "";
    }

    // Parse JSON: {"text": "...", "error": "...", ...}
    size_t pos = output.find("\"text\"");
    if(pos == std::string::npos) return "";
    pos = output.find(':', pos);
    if(pos == std::string::npos) return "";
    pos++;
    while(pos < output.size() && (output[pos] == ' ' || output[pos] == '"')) pos++;
    size_t end = output.find('"', pos);
    if(end == std::string::npos) return "";
    return output.substr(pos, end - pos);
}

static std::string transcribe_via_helper_bin_stream(const std::string& helper_bin,
    const std::string& model_dir, const std::string& device, const std::string& wav){
    char dev_buf[64];
    snprintf(dev_buf, sizeof(dev_buf), "%s", device.c_str());

    std::string cmd = helper_bin +
        " --model-dir " + model_dir +
        " --device " + dev_buf +
        " --audio " + wav +
        " --stream" +
        " --chunk-ms 320" +
        " --left-ms 5600" +
        " --chunk-ctx-ms 1040" +
        " --right-ms 1040" +
        " 2>/dev/null";

    FILE* p = popen(cmd.c_str(), "r");
    if(!p) {
        fprintf(stderr, "[asr-stream] popen failed: %s\n", strerror(errno));
        return "";
    }
    std::string output;
    char buf[4096];
    while(fgets(buf, sizeof(buf), p)) output += buf;
    int rc = pclose(p);
    if(rc != 0) {
        fprintf(stderr, "[asr-stream] helper exited %d\n", rc);
        return "";
    }
    // Parse the transcript from the helper's LAST JSON line. The output is
    // line-delimited {"text":"...","is_final":...} and may contain empty
    // incremental lines first — the naive first-"text"-occurrence parse
    // trips over escapes and empty texts ("," garbage). Walk the lines and
    // keep the longest text seen; it is the final hypothesis.
    try {
        std::string best;
        size_t start = 0;
        while(start < output.size()){
            size_t nl = output.find('\n', start);
            std::string line = output.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
            size_t pos = line.find("\"text\":\"");
            if(pos != std::string::npos){
                pos += 8;
                std::string text;
                for(size_t i = pos; i < line.size(); i++){
                    if(line[i] == '\\' && i + 1 < line.size()){
                        char nx = line[i + 1];
                        switch(nx){
                            case '"': text += '"'; i++; break;
                            case '\\': text += '\\'; i++; break;
                            case 'n': text += '\n'; i++; break;
                            case 'r': text += '\r'; i++; break;
                            case 't': text += '\t'; i++; break;
                            default: text += line[i]; break;
                        }
                    } else if(line[i] == '"'){
                        break;
                    } else {
                        text += line[i];
                    }
                }
                if(text.size() > best.size()) best = text;
            }
            if(nl == std::string::npos) break;
            start = nl + 1;
        }
        if(best.empty()) fprintf(stderr, "[asr-stream] helper produced no text\n");
        return best;
    } catch(...) {
        return "";
    }
}

// NOTE: transcripts paste byte-identical to model output — no punctuation
// or spacing post-processing. An earlier finish_transcript() experiment
// (append ". "/danda on manual stop) fought models that self-punctuate
// ("?" + added "." = "? .") and was removed per user request.
static void copy_clipboard(const std::string& s){
    if(s.empty()) return;
    FILE* p=popen("wl-copy","w");
    fwrite(s.data(),1,s.size(),p); pclose(p);
}
static void type_text(const std::string& s){
    if(s.empty()) return;
    FILE* p=popen("wtype -","w");
    fwrite(s.data(),1,s.size(),p); pclose(p);
}

// ─── live-bar shared state ─────────────────────────
// The popup (RecorderPopup.qml) polls these files at 800ms and renders the
// status. status.json already drives the top-bar icon; we EXTEND it with a
// few fields the popup needs and route the raw mic level through a separate
// `level` file so a high-rate writer never competes with status.json.
//
//   status.json: {"class": "recording"|"transcribing", "stream":1?, "since":epoch}
//     stream   = 1 while REAL-TIME streaming (parakeet-stream). Live phase
//                reports class "recording" + stream (timer+bars in popup);
//                post-tap drain reports "transcribing" (+stream till final).
//                Popup and bar pick glyphs by class + stream.
//     since    = record/stream START epoch (seconds) so the timer can count
//                even if the daemon restarts or the popup mounts late.
//   level:       plain decimal RMS (0.0 .. 1.0), written ~8x/sec by the tee
//                child. Not JSON — a tiny write is a tiny race window.
//   cancel-requested: the popup TOUCHES (creates+updates) this empty file to
//                ask for an abort. The daemon polls it every 200ms.
// ────────────────────────────────────────────────────────────────────

static const std::string g_cancel_file = "/home/shlok/.local/share/npu-asr/state/cancel-requested";
static const std::string g_level_file  = "/home/shlok/.local/share/npu-asr/state/level";
static std::atomic<bool> g_cancel_seen{false};

// Request time of the current take. 0 when idle. The bar/popup use this as
// `since` so the timer counts from the exact tap instant.
static std::atomic<time_t> g_take_start{0};

// write state JSON for bar visibility (mic icon shows when class != idle)
static void write_state(const std::string& cls, const std::string& path){
    if(path.empty()) return;
    std::ofstream f(path);
    f<<"{\"alt\":\""<<cls<<"\",\"class\":\""<<cls<<"\",\"tooltip\":\"\"}\n";
    f.flush();
    // Signal waybar to refresh immediately instead of waiting for poll interval
    system("pkill -RTMIN+8 waybar 2>/dev/null");
}

// Extended state: like write_state but also carries stream + since so the
// popup can tell streaming from plain transcribing and count from take start.
static void write_state_full(const std::string& cls, const std::string& path,
                             bool stream, time_t since){
    if(path.empty()) return;
    std::ofstream f(path);
    f << "{\"alt\":\"" << cls << "\",\"class\":\"" << cls << "\",\"tooltip\":\"\"";
    if(stream)     f << ",\"stream\":1";
    if(since > 0)  f << ",\"since\":" << since;
    f << "}\n";
    f.flush();
    system("pkill -RTMIN+8 waybar 2>/dev/null");
}

// Has the user asked to cancel via the popup's X button? A touch creates the
// file with a monotonically-increasing mtime; we latch it once so the abort
// path and paste gates agree, then clear it so a later take starts fresh.
static bool cancel_requested(){
    if(g_cancel_seen.load()) return true;
    if(::access(g_cancel_file.c_str(), F_OK) == 0){
        g_cancel_seen.store(true);
        fprintf(stderr, "[dictate] cancel requested via popup X\n"); fflush(stderr);
        return true;
    }
    return false;
}

static void clear_cancel(){
    g_cancel_seen.store(false);
    ::unlink(g_cancel_file.c_str());
}

// Best-effort level write from the tee child (streaming) or the record child
// (non-streaming). Called on raw PCM frames; writes a single line. Throttled
// to ~8 writes/sec by only emitting when >=125ms has passed.
//
// NOTE: throttle on WALL time (steady_clock), not clock()/CPU time — the
// fork-let tee children are pure I/O (read pipe -> sanitize -> write file),
// so their clock() advances far slower than wall time and level updates
// would collapse to a trickle (verified: one write per ~10-20s).
static void write_level_rms(const float* f, size_t n){
    if(n == 0) return;
    using clock_t = std::chrono::steady_clock;
    static auto last_emit = clock_t::time_point{};
    auto now = clock_t::now();
    if(now - last_emit < std::chrono::milliseconds(125)) return;
    last_emit = now;
    double sum = 0;
    for(size_t i = 0; i < n; i++){
        float v = f[i];
        if(v < -1.0f) v = -1.0f;
        if(v > 1.0f) v = 1.0f;
        sum += double(v) * double(v);
    }
    double rms = std::sqrt(sum / double(n));
    std::ofstream lf(g_level_file);
    lf << rms;
    lf.flush();
}

// Stamp a valid 44-byte f32 WAV header onto a raw PCM file (matches the
// streaming tee's post-fixup). Used by the non-stream record path, whose tee
// reserves the header then appends sanitized f32 like the streaming tee.
static void fix_wav_header_f32(const std::string& path){
    FILE* f = fopen(path.c_str(), "r+b");
    if(!f) return;
    fseek(f, 0, SEEK_END);
    long total = ftell(f);
    long pcm_bytes = total > 44 ? total - 44 : 0;
    rewind(f);
    uint32_t chunk_size = (uint32_t)(36 + pcm_bytes);
    uint32_t sub2_size = (uint32_t)pcm_bytes;
    uint32_t sample_rate = 16000;
    uint16_t channels = 1, bits = 32, format = 3; /* float */
    uint32_t byte_rate = sample_rate * channels * 4;
    uint16_t block_align = channels * 4;
    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t sub1 = 16; fwrite(&sub1, 4, 1, f);
    fwrite(&format, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&sub2_size, 4, 1, f);
    fclose(f);
    fprintf(stderr, "[dictate] capture.wav header fixed (%ld pcm bytes)\n", pcm_bytes); fflush(stderr);
}

// ---------------- capture buffer (VAD-aware) ----------------
// Replaces pw-record with an in-memory capture loop that runs Silero VAD on
// each 512-sample (32ms) frame. Speech-only frames are collected; recording
// stops automatically after detecting sustained silence (hangover tail),
// eliminating hallucination on silence.
struct CaptureBuffer {
    std::vector<float> samples;       // all captured samples (post-VAD filter)
    bool capturing;
    std::atomic<bool> should_stop;

    CaptureBuffer() : capturing(false) { should_stop.store(false); }

    void reset() {
        samples.clear();
        capturing = false;
        should_stop.store(false);
    }
};

// ALSA/PipeWire capture callback — fills CaptureBuffer with raw audio
// then VAD processes in 32ms chunks
static void capture_callback(const float* data, size_t frames, void* userdata);

// ---------------- daemon (Handy-style) ----------------
// Stays alive but does NOT keep models loaded. Only monitors for the
// Copilot button toggle via SIGUSR1, then does a load -> record -> transcribe -> unload cycle.
static std::atomic<bool> g_toggle{false};
static std::atomic<pid_t> g_rt_child_pid{0};
static std::atomic<pid_t> g_vad_monitor_pid{0}; // track VAD monitoring child for gap detection    // RT streaming child (parakeet-stream)
static std::atomic<pid_t> g_pw_record_pid{0};  // track the pw-record child
static std::atomic<bool> g_rt_stop_signal{false}; // set on 2nd tap during RT streaming
static std::atomic<bool> g_rec_end{false};
static std::atomic<bool> g_transcribing{false}; // true during tail+transcribe: Copilot taps ignored

// ─── Real-time streaming for parakeet-stream ───
// True real-time streaming: pw-record stdout pipes directly to
// parakeet_stream_transcribe_rt stdin. The helper feeds audio chunks to
// libtranscribe and emits incremental JSON {"text":"...","is_final":false/true}
// lines. The daemon forks a child to run the pipeline; the child types text
// as it arrives. On 2nd tap, the daemon sends SIGUSR1 to the child to stop.

// Rolling archive of the last 5 raw streaming takes: each
// stop rotates stream{N}.wav -> stream{N+1}.wav and stores the fresh take as
// stream1.wav, plus the matching live-streamed transcript as
// stream1.txt (etc). Raw mic bytes, untouched — for the mic-vs-pipeline
// self-test (retranscribe with whisper/NPU and diff against parakeet live).
static void archive_stream_take(const std::string& stbase,
                                const std::string& wav,
                                const std::string& transcript){
    for(int i = 4; i >= 1; i--){
        std::string older = stbase + "/stream" + std::to_string(i) + ".wav";
        std::string newer = stbase + "/stream" + std::to_string(i + 1) + ".wav";
        ::rename(older.c_str(), newer.c_str());
        std::string older_t = stbase + "/stream" + std::to_string(i) + ".txt";
        std::string newer_t = stbase + "/stream" + std::to_string(i + 1) + ".txt";
        ::rename(older_t.c_str(), newer_t.c_str());
    }
    if(::access(wav.c_str(), R_OK) == 0){
        std::string dst = stbase + "/stream1.wav";
        // Copy (not rename): the 2nd-tap path may still fall back to the
        // stream.wav file after we return.
        FILE* in = fopen(wav.c_str(), "rb");
        FILE* out = fopen(dst.c_str(), "wb");
        if(in && out){
            char buf[65536];
            size_t n;
            while((n = fread(buf, 1, sizeof(buf), in)) > 0)
                fwrite(buf, 1, n, out);
        }
        if(in) fclose(in);
        if(out) fclose(out);
        std::string txt = stbase + "/stream1.txt";
        FILE* tf = fopen(txt.c_str(), "w");
        if(tf){ fwrite(transcript.data(), 1, transcript.size(), tf); fclose(tf); }
        fprintf(stderr, "[asr-rt] archived take -> stream1.wav\n"); fflush(stderr);
    }
}

static bool read_line_nb(int fd, std::string& line, int timeout_ms){
    line.clear();
    while(true){
        fd_set rfds;
        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        int sel = select(fd + 1, &rfds, NULL, NULL, &tv);
        if(sel <= 0) return false;
        char c;
        ssize_t n = read(fd, &c, 1);
        if(n <= 0) return !line.empty();
        if(c == '\n') return true;
        line += c;
    }
}

static std::string parse_rt_json(const std::string& line, bool& is_final_out){
    is_final_out = (line.find("\"is_final\":true") != std::string::npos);
    size_t pos = line.find("\"text\":\"");
    if(pos == std::string::npos) return "";
    pos += 8;
    std::string text;
    for(size_t i = pos; i < line.size(); i++){
        if(line[i] == '\\' && i + 1 < line.size()){
            char next = line[i + 1];
            switch(next){
                case '"': text += '"'; i++; break;
                case '\\': text += '\\'; i++; break;
                case 'n': text += '\n'; i++; break;
                case 'r': text += '\r'; i++; break;
                case 't': text += '\t'; i++; break;
                case 'u': text += '?'; i++; break;
                default: text += line[i]; break;
            }
        } else if(line[i] == '"'){
            break;
        } else {
            text += line[i];
        }
    }
    return text;
}

static void on_rt_stop(int){
    g_rt_stop_signal.store(true);
}

// Grow a pipe's kernel buffer so a slow-reader stall (e.g. a big first
// mic burst arriving before the model finishes loading) can't fill the
// pipe and make pw-record block or drop early audio. Best-effort: pipe
// still works at default size if the grow fails.
static void widen_pipe(int fd){
    long cur = fcntl(fd, F_GETPIPE_SZ);
    if(cur < 0) return;
    if(cur < 1048576){
        if(fcntl(fd, F_SETPIPE_SZ, 1048576) < 0)
            fprintf(stderr, "[asr-rt] pipe buffer stays %ld\n", cur);
    }
}

static std::string run_rt_stream(const std::string& helper_bin,
                                  const std::string& model_dir,
                                  const std::string& device){
    g_rt_stop_signal.store(false);

    struct sigaction sa; memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_rt_stop; sigaction(SIGUSR1, &sa, NULL);

    int pipe_a[2];
    if(pipe(pipe_a) < 0){
        fprintf(stderr, "[asr-rt] pipe() failed\n");
        return "";
    }
    // 1MB on the mic->helper pipe: covers ~4s of f32@16k while the model
    // loads, so words spoken during load aren't dropped (start-loss fix).
    widen_pipe(pipe_a[0]);
    widen_pipe(pipe_a[1]);

    // Streaming has NO record limit: pw-record runs unbounded and the only
    // exits are the 2nd tap, the 15s silence auto-stop, or daemon shutdown.
    // (Non-streaming models keep their timeout cap in their own fork below.)
    // ALSO tee raw mic to a WAV file: if the live streaming hypothesis is
    // empty (silence/VAD), the 2nd-tap path falls back to transcribing this
    // file instead of pasting nothing.
    std::string tee_wav = "/home/shlok/.local/share/npu-asr/state/stream.wav";
    ::unlink(tee_wav.c_str());

    // JSON pipe for helper -> parent incremental lines.
    int pipe_b[2];
    if(pipe(pipe_b) < 0){
        fprintf(stderr, "[asr-rt] pipe() failed\n");
        close(pipe_a[0]);
        close(pipe_a[1]);
        return "";
    }

    // Fork RT helper FIRST, before pw-record: the helper's tap-instant
    // spool thread must be draining pipe_a from process start, so mic audio
    // is buffered in RAM from the tap instant even while the model loads.
    pid_t helper_pid = fork();
    if(helper_pid == 0){
        dup2(pipe_a[0], STDIN_FILENO);
        close(pipe_a[0]);
        close(pipe_a[1]);
        dup2(pipe_b[1], STDOUT_FILENO);
        close(pipe_b[1]);
        close(pipe_b[0]);
        int dn = open("/dev/null", O_WRONLY);
        if(dn >= 0) dup2(dn, STDERR_FILENO);
        char dev_buf[64];
        snprintf(dev_buf, sizeof(dev_buf), "%s", device.c_str());
        execlp(helper_bin.c_str(), helper_bin.c_str(),
               "--model-dir", model_dir.c_str(),
               "--device", dev_buf,
               "--chunk-ms", "320",
               "--left-ms", "5600",
               "--chunk-ctx-ms", "1040",
               "--right-ms", "1040",
               // Live silence auto-stop (streaming only): 10s of continuous
               // sub-threshold audio ends the session by itself. Armed only
               // after the first speech chunk, so a slow start never
               // triggers it. Other models never pass this, so their
               // behavior is unchanged.
               "--silence-stop-s", "10",
               "--silence-rms", "0.01",
               (char*)NULL);
        _exit(127);
    }
    close(pipe_a[0]);
    close(pipe_b[1]);
    if(helper_pid < 0){
        fprintf(stderr, "[asr-rt] helper fork failed\n");
        close(pipe_a[1]);
        close(pipe_b[0]);
        return "";
    }

    // THEN fork pw-record: by the time the first mic byte lands in pipe_a,
    // the helper's spool thread is already draining it.
    pid_t pw_pid = fork();
    if(pw_pid == 0){
        // pw-record -> tee: stdout pipe goes to BOTH the helper (via pipe_a)
        // and the fallback WAV file.
        int wav_fd = open(tee_wav.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        int tee_pipe[2];
        if(pipe(tee_pipe) < 0) _exit(127);
        // Symmetric 1MB with pipe_a: a disk stall in the tee WAV write needs
        // >~4s to backpressure into the live stream.
        widen_pipe(tee_pipe[0]);
        widen_pipe(tee_pipe[1]);
        pid_t tee_pid = fork();
        if(tee_pid == 0){
            // tee process: stdin <- pw-record, stdout -> pipe_a + wav file
            close(tee_pipe[1]);
            dup2(tee_pipe[0], STDIN_FILENO);
            close(tee_pipe[0]);
            dup2(pipe_a[1], STDOUT_FILENO);
            close(pipe_a[1]);
            close(pipe_b[0]);
            if(wav_fd >= 0){
                // WAV header will be fixed up by the parent after stop;
                // here just append raw PCM after reserving header space.
                // Sanitize NaN/Inf samples to 0.0 first: PipeWire's first
                // startup buffers can carry one (9-Sep self-test: sample 2
                // was NaN in all 5 takes). Both models treat it as speech:
                // streaming Parakeet stays glued to the head garbage and
                // drops the first real sentence; whisper's chunker inflates
                // head RMS to ~0.06 and cuts the first chunk mid-word.
                // Same sanitized bytes go to helper AND wav so both paths
                // see identical audio.
                const char zero[44] = {0};
                (void)!write(wav_fd, zero, sizeof(zero));
                char buf[65536];
                ssize_t n;
                while((n = read(STDIN_FILENO, buf, sizeof(buf))) > 0){
                    size_t nf = (size_t)n / sizeof(float);
                    float* f = (float*)buf;
                    for(size_t k = 0; k < nf; k++){
                        float v = f[k];
                        // !(v <= 1.0f && v >= -1.0f): true for NaN, +Inf,
                        // -Inf AND out-of-range garbage — one branchless-ish
                        // test, no math.h dependency in this fork child.
                        if(!(v <= 1.0f && v >= -1.0f)) f[k] = 0.0f;
                    }
                    write_level_rms(f, nf);
                    (void)!write(STDOUT_FILENO, buf, (size_t)n);
                    (void)!write(wav_fd, buf, (size_t)n);
                }
                close(wav_fd);
            } else {
                char buf[65536];
                ssize_t n;
                while((n = read(STDIN_FILENO, buf, sizeof(buf))) > 0)
                    (void)!write(STDOUT_FILENO, buf, (size_t)n);
            }
            _exit(0);
        }
        close(tee_pipe[0]);
        dup2(tee_pipe[1], STDOUT_FILENO);
        close(tee_pipe[1]);
        close(pipe_a[1]);
        close(pipe_b[0]);
        if(wav_fd >= 0) close(wav_fd);
        int dn = open("/dev/null", O_WRONLY);
        if(dn >= 0) dup2(dn, STDERR_FILENO);
        // NO timeout(1): streaming is unbounded by design. Ends via 2nd tap
        // (SIGTERM below), 15s silence self-stop, or daemon shutdown.
        execlp("pw-record","pw-record","--rate","16000","--channels","1",
               "--format","f32","--channel-map","mono","-",
               (char*)NULL);
        _exit(127);
    }
    close(pipe_a[1]);
    if(pw_pid < 0){
        fprintf(stderr, "[asr-rt] pw-record fork failed\n");
        kill(helper_pid, SIGTERM);
        waitpid(helper_pid, nullptr, 0);
        close(pipe_b[0]);
        return "";
    }

    fprintf(stderr, "[asr-rt] pipeline started (pw_pid=%d, helper_pid=%d)\n", pw_pid, helper_pid); fflush(stderr);

    // Accumulate the latest hypothesis silently in the background.
    // NOTHING is typed or pasted here: 2nd tap copies the final text
    // to the clipboard and pastes it in one BAM shot.
    std::string latest;
    std::string line;
    std::string final_text;

    // Full-drain EOF handshake: every fed frame must be decoded before the
    // final paste, otherwise the tail of the utterance is cut mid-sentence.
    // On 2nd tap the daemon first lets the 1s audio tail through, then
    // signals this loop via SIGUSR1. That signal ONLY closes pw-record's
    // stdin here (EOF through tee -> helper); the helper is never SIGTERM'd
    // or SIGKILL'd — it finalizes on its own once it has drained every
    // frame, so final_text always covers the whole utterance.
    // The helper ALSO exits by itself on the 15s silence auto-stop (emits
    // is_final after finalizing the quiet tail). Either way its exit is
    // handled identically below. Streaming has no record cap at all:
    // pw-record runs unbounded, so pw-record death here is always
    // unexpected — but still must NOT end this loop on its own, since
    // audio may be buffered in the helper's context. Only the helper's
    // is_final line (or its exit) ends the loop.
    bool pw_stdin_closed = false;
    while(true){
        if(g_rt_stop_signal.load() && !pw_stdin_closed){
            fprintf(stderr, "[asr-rt] stop signal received, closing mic input (EOF), waiting for helper drain\n"); fflush(stderr);
            kill(pw_pid, SIGTERM);
            pw_stdin_closed = true;
        }

        if(!read_line_nb(pipe_b[0], line, 100)){
            int status;
            pid_t exited = waitpid(helper_pid, &status, WNOHANG);
            if(exited > 0) break;
            if(pw_stdin_closed){
                // Mic is closed; the helper just needs decode wall-time to
                // drain. Linger here — its is_final line below ends the loop.
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            continue;
        }

        if(line.empty()) continue;

        bool is_final = false;
        std::string text = parse_rt_json(line, is_final);
        if(text.empty()) continue;

        if(is_final){
            final_text = text;
            fprintf(stderr, "[asr-rt] final text: %s\n", final_text.c_str()); fflush(stderr);
            break;
        } else {
            latest = text;
        }
    }

    // Non-destructive shutdown: pw-record gets SIGTERM first (lets the tee
    // flush buffered PCM); only SIGKILL if it refuses to exit. The helper
    // already emitted is_final (or exited) above — never killed mid-decode.
    kill(pw_pid, SIGTERM);
    for(int i = 0; i < 100; i++){
        int status;
        if(waitpid(pw_pid, &status, WNOHANG) > 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if(kill(pw_pid, 0) == 0){
        kill(pw_pid, SIGKILL);
        waitpid(pw_pid, nullptr, 0);
    } else {
        waitpid(pw_pid, nullptr, 0);
    }
    int hstatus;
    if(waitpid(helper_pid, &hstatus, WNOHANG) <= 0){
        // Helper still alive without is_final (shouldn't happen): give it a
        // grace window to finalize on EOF before force-killing.
        for(int i = 0; i < 250; i++){
            if(waitpid(helper_pid, &hstatus, WNOHANG) > 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if(kill(helper_pid, 0) == 0){
            kill(helper_pid, SIGKILL);
            waitpid(helper_pid, nullptr, 0);
        }
    }
    // Reap the tee grandchild (child of pw_pid's process group lineage —
    // actually our grandchild via double-fork, may already be adopted).
    while(waitpid(-1, nullptr, WNOHANG) > 0) {}

    // Fix up the tee'd raw file into a valid WAV (44-byte header + f32 PCM)
    // so the 2nd-tap path can fall back to file transcription.
    {
        std::string tee_wav = "/home/shlok/.local/share/npu-asr/state/stream.wav";
        FILE* f = fopen(tee_wav.c_str(), "r+b");
        if(f){
            fseek(f, 0, SEEK_END);
            long total = ftell(f);
            long pcm_bytes = total > 44 ? total - 44 : 0;
            long samples = pcm_bytes / 4;
            rewind(f);
            uint32_t chunk_size = (uint32_t)(36 + pcm_bytes);
            uint32_t sub2_size = (uint32_t)pcm_bytes;
            uint32_t sample_rate = 16000;
            uint16_t channels = 1, bits = 32, format = 3 /* float */;
            uint32_t byte_rate = sample_rate * channels * 4;
            uint16_t block_align = channels * 4;
            fwrite("RIFF", 1, 4, f);
            fwrite(&chunk_size, 4, 1, f);
            fwrite("WAVE", 1, 4, f);
            fwrite("fmt ", 1, 4, f);
            uint32_t sub1 = 16; fwrite(&sub1, 4, 1, f);
            fwrite(&format, 2, 1, f);
            fwrite(&channels, 2, 1, f);
            fwrite(&sample_rate, 4, 1, f);
            fwrite(&byte_rate, 4, 1, f);
            fwrite(&block_align, 2, 1, f);
            fwrite(&bits, 2, 1, f);
            fwrite("data", 1, 4, f);
            fwrite(&sub2_size, 4, 1, f);
            fclose(f);
            fprintf(stderr, "[asr-rt] tee wav fixed up: %ld samples\n", samples); fflush(stderr);
        }
    }

    // Longest-wins: the loop above may return via helper exit without an
    // is_final line, or the live view may have frozen mid-utterance while
    // the model kept decoding — whichever of the final line and the last
    // incremental hypothesis is longer is the fuller transcript.
    if(final_text.size() < latest.size()) final_text = latest;
    // Rolling archive: keep the raw take + live transcript
    // BEFORE the daemon unlinks stream.wav below. Copy, not rename — the
    // 2nd-tap fallback path may still need stream.wav after we return.
    archive_stream_take("/home/shlok/.local/share/npu-asr/state",
                        "/home/shlok/.local/share/npu-asr/state/stream.wav",
                        final_text);
    return final_text;
}

static void on_sigchld(int){
    // Only set g_rec_end for the actual record-child exit, not for
    // the double-fork audio-player children. In streaming mode we wake
    // the loop on any child death (grandchildren included) but DON'T
    // reap — the loop's waitpid(WNOHANG) must see the RT child itself.
    // In non-streaming mode we reap children via waitpid(-1) and only
    // set g_rec_end if the reaped PID matches g_pw_record_pid.
    pid_t rt_pid  = g_rt_child_pid.load();
    pid_t pw_pid  = g_pw_record_pid.load();
    pid_t vm_pid  = g_vad_monitor_pid.load();

    if(rt_pid > 0){
        // Streaming: just wake the loop; it will inspect the RT child
        // with waitpid(WNOHANG) and distinguish grandchild deaths.
        g_rec_end.store(true);
    } else if(pw_pid > 0){
        // Non-streaming: reap any direct child that exited, but only
        // flag g_rec_end if it's actually the pw-record child.
        int status;
        pid_t reaped;
        while((reaped = waitpid(-1, &status, WNOHANG)) > 0){
            if(reaped == pw_pid)
                g_rec_end.store(true);
        }
    }
    if(vm_pid > 0) kill(vm_pid, SIGTERM);
}

// ─── daemon (Handy-style) ─────────
// Stays alive but does NOT keep models loaded. Only monitors for the
// Copilot button toggle via SIGUSR1, then does a load -> record -> transcribe -> unload cycle.
static void on_sigusr1(int){ g_toggle.store(true); }

static void run_daemon(){
    std::string stbase="/home/shlok/.local/share/npu-asr/state";
    std::string stfile=stbase+"/status.json";
    std::string pidfile=stbase+"/daemon.pid";
    mkdir(stbase.c_str(),0755);

    std::ofstream(pidfile)<<getpid()<<"\n";

    struct sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=on_sigusr1; sigaction(SIGUSR1,&sa,NULL);

    struct sigaction sch; memset(&sch,0,sizeof(sch));
    sch.sa_handler=on_sigchld; sigaction(SIGCHLD,&sch,NULL);

    write_state("idle", stfile);
    fprintf(stderr,"[dictate] daemon ready (load-on-demand). Tap Copilot to start.\n");

    int rec_running=0;
    bool stream_mode = false;
    for(;;){
        // Poll with a 100ms timeout instead of a blocking pause() so the
        // loop can notice a popup cancel-request even when no signal arrives,
        // and so taps are processed promptly (the popup polls at 100ms too).
        {
            fd_set rfds; FD_ZERO(&rfds);
            struct timeval tv; tv.tv_sec = 0; tv.tv_usec = 100000;
            select(0, &rfds, NULL, NULL, &tv);
        }
        // Cancellation is checked BEFORE the signal flags, at every wake: a
        // cancel that lands during an active take aborts it regardless of
        // state, including mid-transcribe (g_transcribing) — the post-paste
        // window is the only truly-too-late case, handled by the paste gates.
        if(rec_running && cancel_requested()){
            g_toggle.store(false);
            g_cancel_seen.store(true);
            if(stream_mode){
                // Abort streaming: kill the RT child (which owns pw-record +
                // helper) and its lineage, then clear temps without paste.
                pid_t rt_pid = g_rt_child_pid.exchange(0);
                if(rt_pid > 0){
                    fprintf(stderr, "[dictate] cancel: terminating RT child %d\n", rt_pid); fflush(stderr);
                    kill(rt_pid, SIGTERM);
                    for(int i = 0; i < 30; i++){
                        int st; if(waitpid(rt_pid, &st, WNOHANG) > 0) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                    if(kill(rt_pid, 0) == 0){ kill(rt_pid, SIGKILL); waitpid(rt_pid, nullptr, 0); }
                }
                pid_t pw = g_pw_record_pid.exchange(0);
                if(pw > 0){ kill(pw, SIGKILL); waitpid(pw, nullptr, 0); }
            } else {
                // Abort non-streaming: kill pw-record (wrapped by the timeout
                // child) and the tee lineage.
                pid_t pw = g_pw_record_pid.exchange(0);
                if(pw > 0){
                    fprintf(stderr, "[dictate] cancel: terminating record child %d\n", pw); fflush(stderr);
                    kill(pw, SIGTERM);
                    for(int i = 0; i < 50; i++){
                        int st; if(waitpid(pw, &st, WNOHANG) > 0) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    if(kill(pw, 0) == 0){ kill(pw, SIGKILL); waitpid(pw, nullptr, 0); }
                }
            }
            // Clean temporary audio + level + cancel state.
            ::unlink((stbase + "/stream.wav").c_str());
            ::unlink((stbase + "/rt_final.txt").c_str());
            ::unlink((stbase + "/capture.wav").c_str());
            ::unlink(g_level_file.c_str());
            clear_cancel();
            g_transcribing.store(false);
            write_state("idle", stfile);
            rec_running = 0;
            stream_mode = false;
            fprintf(stderr, "[dictate] cancelled, discarded take\n"); fflush(stderr);
            continue;
        }

        bool tap = g_toggle.exchange(false);
        bool rec_ended = g_rec_end.exchange(false);
        if(!tap && !rec_ended) continue;
        if(rec_ended && !tap && rec_running && !stream_mode)
            fprintf(stderr,"[dictate] recording window hit its cap, auto-transcribing\n"), fflush(stderr);

        // Guard: a tap arriving while the tail+transcribe phase runs must
        // not disturb it — drop the tap, keep transcribing, still paste.
        if(tap && g_transcribing.load()){
            tap = false;
            fprintf(stderr, "[dictate] tap ignored: transcription in progress\n"); fflush(stderr);
        }

        // Cap path (non-streaming only): timeout(1) killed pw-record at the
        // record limit. 1s audio tail first — the user may have been
        // mid-word when the cap hit and only stops after noticing the mic
        // icon is gone. Same stop+transcribe handling as the 2nd tap below;
        // a tap during that transcribe is ignored by the g_transcribing
        // guard, never destructive. Streaming deaths flow through the RT
        // branch below instead, so they don't trip the non-streaming cap.
        if(rec_ended && !tap && rec_running && !stream_mode){
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            tap = true;
            rec_ended = false;
        }
        // Clear any stale rec_ended flag in streaming mode; the RT branch
        // below uses waitpid(WNOHANG) to tell grandchild deaths apart.
        if(stream_mode && rec_ended) rec_ended = false;

        // RT child exited on its own: the 15s silence auto-stop fired. The
        // helper already finalized and wrote rt_final.txt; the 1s audio tail
        // below lets any trailing word fragment flush before pasting.
        // A tap landing here must be dropped (g_transcribing guard), never
        // destructive — same rule as every other transcribe phase.
        // NOTE: streaming has no record cap, so ANY self-exit here IS the
        // silence trigger — there is no timeout-cap path for streaming.
        if(!tap && rec_running && stream_mode){
            pid_t rt_pid = g_rt_child_pid.load();
            if(rt_pid > 0){
                int status;
                pid_t w = waitpid(rt_pid, &status, WNOHANG);
                if(w == 0){
                    // RT child still running (this wakeup was a stray
                    // grandchild SIGCHLD) — stay in streaming, keep waiting.
                    continue;
                }
                if(w > 0){
                    fprintf(stderr, "[dictate] RT child self-exited (silence auto-stop)\n"); fflush(stderr);
                    // audio-feedback: recording stopped (15s silence auto-stop)
                    play_stop_sound();
                    g_rt_child_pid.store(0);
                    g_transcribing.store(true);
                    // 1s audio tail: trailing word fragment flushes before
                    // the paste, same as the 2nd-tap path below. No mic to
                    // stop here (helper already exited, pw-record reaped by
                    // the RT child); this is pure settle time before reading
                    // rt_final.txt.
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    std::string rt_final_file = stbase + "/rt_final.txt";
                    std::string transcript;
                    std::ifstream rt_ifs(rt_final_file);
                    if(rt_ifs.good()){
                        std::string all((std::istreambuf_iterator<char>(rt_ifs)),
                                         std::istreambuf_iterator<char>());
                        rt_ifs.close();
                        std::remove(rt_final_file.c_str());
                        transcript = all;
                    }
                    if(transcript.empty()){
                        std::string tee_wav = stbase + "/stream.wav";
                        if(::access(tee_wav.c_str(), R_OK) == 0){
                            SF_INFO ti; memset(&ti, 0, sizeof(ti));
                            SNDFILE* tsf = sf_open(tee_wav.c_str(), SFM_READ, &ti);
                            long frames = tsf ? (long)ti.frames : 0;
                            if(tsf) sf_close(tsf);
                            if(frames > 1600){
                                std::string fb_model = read_model();
                                std::string fb_dir = resolve_model_dir(fb_model);
                                std::string fb_dev = read_device();
                                if(fb_dev == "NPU") fb_dev = "GPU";
                                transcript = transcribe_via_helper_bin_stream(
                                    "/home/shlok/.local/bin/parakeet_stream_transcribe",
                                    fb_dir, fb_dev, tee_wav);
                            }
                        }
                        ::unlink((stbase + "/stream.wav").c_str());
                    } else {
                        ::unlink((stbase + "/stream.wav").c_str());
                    }
                    if(cancel_requested()){
                        fprintf(stderr, "[dictate] cancel during in-flight finalize, discarding (paste not fired)\n"); fflush(stderr);
                        clear_cancel();
                        g_transcribing.store(false);
                        write_state("idle", stfile);
                        rec_running = 0;
                        stream_mode = false;
                        continue;
                    }
                    if(!transcript.empty()){
                        // Paste model output byte-identical — no punctuation
                        // or spacing post-processing (removed per user request).
                        fprintf(stderr, "[dictate] RT auto-exit final: %s\n", transcript.c_str()); fflush(stderr);
                        copy_clipboard(transcript);
                        paste_clipboard();
                    } else {
                        fprintf(stderr, "[dictate] no streaming text produced\n"); fflush(stderr);
                    }
                    g_transcribing.store(false);
                    write_state("idle", stfile);
                    rec_running = 0;
                    stream_mode = false;
                }
            }
            continue;
        }
        fprintf(stderr,"[dictate] toggle: rec_running=%d stream=%d\n", rec_running, (int)stream_mode); fflush(stderr);

        if(rec_running){
            // 2nd tap: stop + transcribe/paste
            if(!read_enabled()){
                fprintf(stderr,"[dictate] disabled, discarding recording\n"); fflush(stderr);
                pid_t rt_pid = g_rt_child_pid.exchange(0);
                if(rt_pid > 0) kill(rt_pid, SIGTERM);
                pid_t pw = g_pw_record_pid.exchange(0);
                if(pw > 0) kill(pw, SIGTERM);
                rec_running = 0;
                stream_mode = false;
                write_state("idle", stfile);
                continue;
            }

            write_state("transcribing", stfile);

            if(stream_mode){
                // RT streaming: send SIGUSR1 to child to finalize.
                // Guard against taps during the tail+finalize below.
                g_transcribing.store(true);
                // Stop sound fires at the TAP instant — before the 1s audio
                // tail — so the user hears the blip immediately. Async playback
                // overlaps the tail with zero timing impact.
                play_stop_sound();
                // 1s audio tail (any model, incl. future ones): the trailing
                // word fragment reaches the helper before it is asked to
                // finalize. 500ms clipped slow word endings; 1s is safe.
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                pid_t rt_pid = g_rt_child_pid.exchange(0);
                if(rt_pid > 0){
                    kill(rt_pid, SIGUSR1);
                    fprintf(stderr, "[dictate] sent stop to RT child (pid=%d)\n", rt_pid); fflush(stderr);
                    int status;
                    for(int i = 0; i < 500; i++){
                        if(waitpid(rt_pid, &status, WNOHANG) > 0) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                    if(kill(rt_pid, 0) == 0){
                        kill(rt_pid, SIGKILL);
                        waitpid(rt_pid, nullptr, 0);
                    }
                }

                // Read final text from state file (read the WHOLE file:
                // transcripts contain spaces/newlines, not just one line)
                std::string rt_final_file = stbase + "/rt_final.txt";
                std::string transcript;
                std::ifstream rt_ifs(rt_final_file);
                if(rt_ifs.good()){
                    std::string all((std::istreambuf_iterator<char>(rt_ifs)),
                                     std::istreambuf_iterator<char>());
                    rt_ifs.close();
                    std::remove(rt_final_file.c_str());
                    transcript = all;
                }
                if(transcript.empty()){
                    // Live hypothesis was empty (silence/too-quiet mic):
                    // fall back to transcribing the tee'd WAV file so the
                    // utterance is not lost.
                    std::string tee_wav = stbase + "/stream.wav";
                    if(::access(tee_wav.c_str(), R_OK) == 0){
                        SF_INFO ti; memset(&ti, 0, sizeof(ti));
                        SNDFILE* tsf = sf_open(tee_wav.c_str(), SFM_READ, &ti);
                        long frames = tsf ? (long)ti.frames : 0;
                        if(tsf) sf_close(tsf);
                        fprintf(stderr, "[dictate] RT empty, fallback file %s (%ld frames)\n",
                                tee_wav.c_str(), frames); fflush(stderr);
                        if(frames > 1600){
                            std::string fb_model = read_model();
                            std::string fb_dir = resolve_model_dir(fb_model);
                            std::string fb_dev = read_device();
                            if(fb_dev == "NPU") fb_dev = "GPU";
                            std::string helper_bin = "/home/shlok/.local/bin/parakeet_stream_transcribe";
                            transcript = transcribe_via_helper_bin_stream(
                                helper_bin, fb_dir, fb_dev, tee_wav);
                        }
                    }
                    ::unlink((stbase + "/stream.wav").c_str());
                } else {
                    ::unlink((stbase + "/stream.wav").c_str());
                }
                if(cancel_requested()){
                    fprintf(stderr, "[dictate] cancel during streaming finalize, discarding\n"); fflush(stderr);
                    clear_cancel();
                    g_transcribing.store(false);
                    write_state("idle", stfile);
                    rec_running = 0;
                    stream_mode = false;
                    continue;
                }
                if(!transcript.empty()){
                    // Paste model output byte-identical — no punctuation or
                    // spacing post-processing (removed per user request).
                    fprintf(stderr, "[dictate] RT final: %s\n", transcript.c_str()); fflush(stderr);
                    copy_clipboard(transcript);
                    paste_clipboard();
                    fprintf(stderr, "[dictate] streaming transcript pasted\n"); fflush(stderr);
                } else {
                    fprintf(stderr, "[dictate] no streaming text produced\n"); fflush(stderr);
                }
                g_transcribing.store(false);
                write_state("idle", stfile);
                rec_running = 0;
                stream_mode = false;
                continue;
            }

            // Non-streaming (any model: whisper, Nepali, Deutsch, ...):
            // guard the tail+transcribe phase against taps.
            g_transcribing.store(true);
            // Stop sound fires at the 2nd-tap / cap-timeout instant — before
            // the 1s audio tail — so the user hears the blip immediately. Async
            // playback overlaps the tail with zero timing impact.
            play_stop_sound();
            // Give the mic 1s to flush the trailing word fragment before
            // killing pw-record — the tail of the last word gets cut
            // otherwise. 500ms clipped slow endings; 1s is safe.
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            pid_t pw_pid = g_pw_record_pid.exchange(0);
            if(pw_pid > 0){
                kill(pw_pid, SIGTERM);
                for(int i = 0; i < 100; i++){
                    int status;
                    if(waitpid(pw_pid, &status, WNOHANG) > 0) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                if(kill(pw_pid, 0) == 0){
                    kill(pw_pid, SIGKILL);
                    waitpid(pw_pid, nullptr, 0);
                }
                fprintf(stderr, "[dictate] pw-record stopped, waiting for file flush...\n"); fflush(stderr);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            std::string wav = stbase + "/capture.wav";
            for(int attempt = 0; attempt < 50; attempt++){
                if(::access(wav.c_str(), R_OK) == 0){
                    SF_INFO check; memset(&check, 0, sizeof(check));
                    if(sf_open(wav.c_str(), SFM_READ, &check) != nullptr){
                        SF_INFO dummy; memset(&dummy, 0, sizeof(dummy));
                        SNDFILE* sf = sf_open(wav.c_str(), SFM_READ, &dummy);
                        if(sf){ sf_close(sf); break; }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if(::access(wav.c_str(), R_OK) != 0){
                fprintf(stderr, "[dictate] no audio file to transcribe\n");
                g_transcribing.store(false);
                write_state("idle", stfile);
                rec_running = 0;
                continue;
            }

            std::string model_name = read_model();
            std::string device = read_device();
            fprintf(stderr, "[dictate] model=\"%s\" device=%s\n", model_name.c_str(), device.c_str());

            std::string model_dir = resolve_model_dir(model_name);
            if(model_name.empty() || model_dir.empty()){
                fprintf(stderr, "[dictate] model.txt \"%s\" not found\n", model_name.c_str());
                g_transcribing.store(false);
                write_state("idle", stfile);
                rec_running = 0;
                continue;
            }

            std::string backend = model_backend(model_name);
            fprintf(stderr, "[dictate] backend=%s for model=%s\n", backend.c_str(), model_name.c_str());

            std::string transcript;

            if(backend == "whisper" || backend == "qwen3" || backend == "indicwav2vec"){
                std::string helper = "/home/shlok/.local/share/npu-asr/whisper_transcribe.py";
                if(backend == "qwen3") helper = "/home/shlok/.local/share/npu-asr/qwen3_transcribe.py";
                if(backend == "indicwav2vec") helper = "/home/shlok/.local/share/npu-asr/indicwav2vec_transcribe.py";
                std::string lang = "";
                if(model_is_multilingual(model_dir)) lang = read_language();
                transcript = transcribe_via_helper(helper, model_dir, device, wav, lang);
            } else if(backend == "parakeet-stream"){
                // Safety fallback — normally handled by RT child
                std::string helper_bin = "/home/shlok/.local/bin/parakeet_stream_transcribe";
                std::string dev_for_helper = device;
                if(dev_for_helper == "NPU") dev_for_helper = "GPU";
                transcript = transcribe_via_helper_bin_stream(helper_bin, model_dir, dev_for_helper, wav);
            } else if(backend == "parakeet-ov"){
                Asr asr;
                std::string enc_model = model_dir + "/encoder-model.int8.onnx";
                std::string dec_model = model_dir + "/decoder_joint-model.int8.onnx";
                std::string vocab_file = model_dir + "/parakeet_v3_vocab.json";
                if(::access(enc_model.c_str(), R_OK) != 0){
                    enc_model = model_dir + "/parakeet_encoder.xml";
                    dec_model = model_dir + "/parakeet_decoder.xml";
                    vocab_file = model_dir + "/parakeet_v3_vocab.json";
                }
                if(::access(enc_model.c_str(), R_OK) == 0){
                    MODEL_DIR = model_dir;
                    asr.load_models(device, T_MAX);
                    SileroVad vad;
                    std::string vad_model_path = VAD_DIR + "/silero_vad_v4.onnx";
                    if(VAD_ENABLED && vad.load(vad_model_path, VAD_THRESHOLD)){
                        long rate=0;
                        auto wavd=read_audio(wav,&rate);
                        if(!wavd.empty()){
                            std::string full_text;
                            std::vector<float> segment;
                            bool was_in_speech = false;
                            size_t offset = 0;
                            vad.reset();
                            while(offset + VAD_FRAME_SAMPLES <= wavd.size()){
                                bool speech_now = vad.is_speech(wavd.data() + offset, VAD_FRAME_SAMPLES);
                                if(speech_now && !was_in_speech) segment.clear();
                                if(speech_now){
                                    segment.insert(segment.end(),
                                        wavd.begin() + offset, wavd.begin() + offset + VAD_FRAME_SAMPLES);
                                }
                                if(!speech_now && was_in_speech){
                                    if(!segment.empty()){
                                        long N_seg = (long)segment.size();
                                        std::string seg_text = asr.transcribe(segment, N_seg, 16000);
                                        if(!seg_text.empty()){
                                            if(!full_text.empty()) full_text += " ";
                                            full_text += seg_text;
                                        }
                                    }
                                    segment.clear();
                                }
                                was_in_speech = speech_now;
                                offset += VAD_FRAME_SAMPLES;
                            }
                            if(was_in_speech && !segment.empty()){
                                long N_seg = (long)segment.size();
                                std::string seg_text = asr.transcribe(segment, N_seg, 16000);
                                if(!seg_text.empty()){
                                    if(!full_text.empty()) full_text += " ";
                                    full_text += seg_text;
                                }
                            }
                            transcript = full_text;
                        }
                    } else {
                        long rate=0;
                        auto wavd=read_audio(wav,&rate);
                        if(!wavd.empty()){
                            long N_real=(long)wavd.size();
                            if(N_real>N_MAX) N_real=N_MAX;
                            transcript = asr.transcribe(wavd, N_real, rate);
                        }
                    }
                }
            }

            if(cancel_requested()){
                // Popup X pressed during non-stream transcribe: best-effort
                // discard. If paste already fired this is too late, but the
                // paste gate below runs immediately before it so this is the
                // common case.
                fprintf(stderr, "[dictate] cancel during non-stream transcribe, discarding (paste not fired)\n"); fflush(stderr);
                clear_cancel();
                ::unlink((stbase + "/capture.wav").c_str());
                ::unlink((stbase + "/level").c_str());
                g_transcribing.store(false);
                write_state("idle", stfile);
                rec_running = 0;
                stream_mode = false;
                continue;
            }

            if(!transcript.empty()){
                // Paste model output byte-identical — no punctuation or
                // spacing post-processing (removed per user request).
                fprintf(stderr,"[dictate] transcript: %s\n", transcript.c_str()); fflush(stderr);
                printf("%s\n", transcript.c_str()); fflush(stdout);
                copy_clipboard(transcript);
                paste_clipboard();
            } else {
                fprintf(stderr,"[dictate] no speech detected\n");
            }

            ::rename(wav.c_str(), (stbase + "/capture.last.wav").c_str());
            g_transcribing.store(false);
            write_state("idle", stfile);
            rec_running = 0;
        } else {
            // 1st tap: start recording
            if(!tap) continue;
            if(!read_enabled()){
                write_state("idle", stfile);
                fprintf(stderr,"[dictate] disabled (enabled=off), ignoring tap\n"); fflush(stderr);
                continue;
            }

            std::string rec_model_name = read_model();
            std::string rec_backend = model_backend(rec_model_name);
            stream_mode = (rec_backend == "parakeet-stream");

            if(stream_mode){
                // REAL-TIME STREAMING: fork RT child that pipes pw-record | helper
                std::string rt_bin = "/home/shlok/.local/bin/parakeet_stream_transcribe_rt";
                std::string model_dir = resolve_model_dir(rec_model_name);
                std::string dev = read_device();
                if(dev == "NPU") dev = "GPU";

                pid_t rt_pid = fork();
                if(rt_pid == 0){
                    signal(SIGCHLD, SIG_DFL);
                    struct sigaction sa_stop; memset(&sa_stop, 0, sizeof(sa_stop));
                    sa_stop.sa_handler = on_rt_stop;
                    sigaction(SIGTERM, &sa_stop, NULL);
                    sigaction(SIGUSR1, &sa_stop, NULL);
                    std::string final_text = run_rt_stream(rt_bin, model_dir, dev);
                    std::string rt_final_file = stbase + "/rt_final.txt";
                    std::ofstream ofs(rt_final_file);
                    if(ofs.good()) ofs << final_text;
                    ofs.close();
                    fprintf(stderr, "[asr-rt-child] wrote rt_final.txt, text='%s'\n", final_text.c_str()); fflush(stderr);
                    _exit(0);
                }
                rec_running = 1;
                g_rt_child_pid.store(rt_pid);
                g_pw_record_pid.store(0);
                g_rec_end.store(false);
                g_take_start.store((time_t)time(nullptr));
                g_cancel_seen.store(false);
                ::unlink(g_cancel_file.c_str());
                ::unlink(g_level_file.c_str());
                // "recording" (not "transcribing") for the live streaming phase:
                // the popup must show the stream glyph + timer + level bars,
                // exactly like non-streaming's recording phase. The existing
                // "transcribing" state below (2nd-tap / silence-finalize) still
                // covers the tail-drain phase.
                write_state_full("recording", stfile, true, g_take_start.load());
                // audio-feedback: mic opening (streaming path)
                play_start_sound();
                fprintf(stderr, "[dictate] RT streaming started (pid=%d)\n", rt_pid); fflush(stderr);
                continue;
            }

            // Non-streaming: standard record-then-transcribe. pw-record pipes
            // through a small tee (like the RT path) so we get live level
            // bars AND a sanitized capture.wav for the fallback transcribe.
            // The tee reserves a 44-byte header; the parent stamps a real
            // WAV header after the kill (same as the streaming tee).
            std::string wav = stbase + "/capture.wav";
            pid_t pid = fork();
            if(pid==0){
                // Outer child: timeout wraps pw-record and a downstream tee.
                int tee_pipe[2];
                if(pipe(tee_pipe) < 0) _exit(127);
                pid_t tee_pid = fork();
                if(tee_pid == 0){
                    // Tee grandchild: sanitize f32, write level + raw PCM.
                    close(tee_pipe[1]);
                    dup2(tee_pipe[0], STDIN_FILENO);
                    close(tee_pipe[0]);
                    int wav_fd = open(wav.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                    if(wav_fd >= 0){
                        const char zero[44] = {0};
                        (void)!write(wav_fd, zero, sizeof(zero));
                        char buf[65536];
                        ssize_t nn;
                        while((nn = read(STDIN_FILENO, buf, sizeof(buf))) > 0){
                            size_t nf = (size_t)nn / sizeof(float);
                            float* f = (float*)buf;
                            for(size_t k = 0; k < nf; k++){
                                float v = f[k];
                                if(!(v <= 1.0f && v >= -1.0f)) f[k] = 0.0f;
                            }
                            write_level_rms(f, nf);
                            (void)!write(wav_fd, buf, (size_t)nn);
                        }
                        close(wav_fd);
                    } else {
                        char buf[65536];
                        ssize_t nn;
                        while((nn = read(STDIN_FILENO, buf, sizeof(buf))) > 0){
                            size_t nf = (size_t)nn / sizeof(float);
                            float* f = (float*)buf;
                            for(size_t k = 0; k < nf; k++){
                                float v = f[k];
                                if(!(v <= 1.0f && v >= -1.0f)) f[k] = 0.0f;
                            }
                            write_level_rms(f, nf);
                        }
                    }
                    // Stamp a valid WAV header now that all PCM is flushed.
                    // Doing it here (in the tee, on EOF side of the pipe)
                    // avoids racing the parent, which only polls for a
                    // readable file afterwards.
                    fix_wav_header_f32(wav);
                    _exit(0);
                }
                close(tee_pipe[0]);
                dup2(tee_pipe[1], STDOUT_FILENO);
                close(tee_pipe[1]);
                int dn = open("/dev/null", O_WRONLY);
                if(dn >= 0) dup2(dn, STDERR_FILENO);
                char timeout_str[32];
                snprintf(timeout_str, sizeof(timeout_str), "%d", (int)MAX_SECS);
                execlp("timeout","timeout", timeout_str,
                       "pw-record","--rate","16000","--channels","1",
                       "--format","f32","--channel-map","mono", "-", (char*)NULL);
                _exit(127);
            }
            rec_running = 1;
            g_pw_record_pid.store(pid);
            g_rec_end.store(false);
            g_take_start.store((time_t)time(nullptr));
            g_cancel_seen.store(false);
            ::unlink(g_cancel_file.c_str());
            ::unlink(g_level_file.c_str());
            write_state_full("recording", stfile, false, g_take_start.load());
            // audio-feedback: mic opening (non-streaming path)
            play_start_sound();
            fprintf(stderr,"[dictate] recording started (pid=%d)\n", pid); fflush(stderr);
        }
    }
}


int main(int argc,char**argv){
    std::string mode="transcribe", wavpath;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--transcribe"||a=="-t") mode="transcribe";
        else if(a=="--type") mode="type";
        else if(a=="--daemon") mode="daemon";
        else if(a=="--device" && i+1<argc){ DEVICE=argv[++i]; }
        else if(a=="--max-secs" && i+1<argc){ MAX_SECS=atof(argv[++i]); N_MAX=(long)(MAX_SECS*16000); T_MAX=std::max<long>(1,1+(N_MAX+512-400)/160); }
        else if(a=="--cache" && i+1<argc){ CACHE_DIR=argv[++i]; }
        else if(a=="--model" && i+1<argc){ MODEL_DIR=argv[++i]; }
        else if(a=="--vad-threshold" && i+1<argc){ VAD_THRESHOLD=(float)atof(argv[++i]); }
        else if(a=="--vad-off" || a=="--no-vad") { VAD_ENABLED=false; }
        else if(!a.empty() && a[0]!='-') wavpath=a;
    }

    if(mode=="daemon"){
        run_daemon();
        return 0;
    }

    if(mode=="transcribe" && wavpath.empty()){
        fprintf(stderr,"dictate: no wav given\n"); return 2;
    }

    if(mode=="transcribe" || mode=="type"){
        Asr asr;
        SileroVad vad;
        std::string vad_model_path = VAD_DIR + "/silero_vad_v4.onnx";

        asr.load_models(DEVICE, T_MAX);

        long rate=0;
        auto wav=read_audio(wavpath,&rate);
        long N_real=(long)wav.size();
        if(N_real>N_MAX) N_real=N_MAX;

        if (VAD_ENABLED && vad.load(vad_model_path, VAD_THRESHOLD)) {
            // Run VAD prefiltering to remove silence-only frames
            std::vector<float> filtered;
            filtered.reserve(N_real);

            size_t offset = 0;
            vad.reset();
            while (offset + VAD_FRAME_SAMPLES <= (size_t)N_real) {
                if (vad.is_speech(wav.data() + offset, VAD_FRAME_SAMPLES)) {
                    filtered.insert(filtered.end(),
                        wav.begin() + offset,
                        wav.begin() + offset + VAD_FRAME_SAMPLES);
                }
                offset += VAD_FRAME_SAMPLES;
            }

            if(!filtered.empty()){
                auto F=nemo::extract(filtered,(long)filtered.size());
                nemo::pad_to(F,T_MAX);
                std::string text=asr.transcribe(F.data,(long)F.T,(long)F.lens);
                printf("%s\n", text.c_str());
                if(mode=="type"){ type_text(text); copy_clipboard(text); }
            } else {
                fprintf(stderr,"[dictate] no speech detected by VAD\n");
                printf("\n");
            }
        } else {
            // No VAD: use original pipeline
            auto F=nemo::extract(wav,N_real);
            nemo::pad_to(F,T_MAX);
            std::string text=asr.transcribe(F.data,(long)F.T,(long)F.lens);
            printf("%s\n", text.c_str());
            if(mode=="type"){ type_text(text); copy_clipboard(text); }
        }
    }
    return 0;
}
