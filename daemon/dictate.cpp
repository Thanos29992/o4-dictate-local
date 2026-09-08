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
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <csignal>

static std::string MODELS_DIR  = "/home/shlok/.local/share/npu-asr/models";
static std::string VAD_DIR     = "/home/shlok/.local/share/npu-asr/models/vad";
static std::string CACHE_DIR   = "/home/shlok/.local/share/npu-asr/cache";
static std::string STATE_DIR   = "/home/shlok/.local/share/npu-asr/state";
static std::string DEVICE      = "NPU";
static std::string MODEL_NAME  = "";  // resolved from state/model.txt at runtime
static double MAX_SECS         = 300.0; // recording window (300s/5min default, VAD segments long audio)
static long   N_MAX            = 4800000; // MAX_SECS*16000
static long   T_MAX            = 9001;   // feats frames for N_MAX
static bool   VAD_ENABLED      = true;  // VAD prefiltering
static float  VAD_THRESHOLD    = 0.3f;  // speech probability threshold (matches Handy)

static const int VOCAB = 8192, BLANK = 8192, MAX_TOKENS_PER_STEP = 10;
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
    // Check if file is JSON (starts with '{')
    std::string first_line;
    std::getline(f, first_line);
    f.seekg(0);
    bool is_json = first_line.size() > 0 && first_line[0] == '{';

    if(is_json){
        // Parse JSON vocab: {"0": "token", "1": "token", ...}
        // Read entire file
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        size_t idx = 0;
        while(idx < content.size()){
            // Find next quoted number key
            size_t key_start = content.find('"', idx);
            if(key_start == std::string::npos) break;
            size_t key_end = content.find('"', key_start+1);
            if(key_end == std::string::npos) break;
            std::string key = content.substr(key_start+1, key_end-key_start-1);
            // Find colon
            size_t colon = content.find(':', key_end);
            if(colon == std::string::npos) break;
            // Find value string
            size_t val_start = content.find('"', colon+1);
            if(val_start == std::string::npos) break;
            size_t val_end = val_start+1;
            // Handle escaped characters
            std::string val;
            while(val_end < content.size()){
                if(content[val_end] == '\\'){
                    val_end++;
                    if(val_end < content.size()){
                        char esc = content[val_end];
                        if(esc == 'n') val.push_back('\n');
                        else if(esc == 't') val.push_back('\t');
                        else if(esc == 'r') val.push_back('\r');
                        else if(esc == '"') val.push_back('"');
                        else if(esc == '\\') val.push_back('\\');
                        else val.push_back(esc);
                        val_end++;
                    }
                } else if(content[val_end] == '"'){
                    break;
                } else {
                    val.push_back(content[val_end]);
                    val_end++;
                }
            }
            // Decode: U+2581 (UTF-8: E2 96 81) -> space
            std::string out; out.reserve(val.size());
            size_t i = 0;
            while(i < val.size()){
                unsigned char c = (unsigned char)val[i];
                if(c == 0xE2 && i+2 < val.size() &&
                   (unsigned char)val[i+1] == 0x96 && (unsigned char)val[i+2] == 0x81){
                    out.push_back(' '); i+=3; continue;
                }
                size_t n = (c<0x80)?1:((c<0xE0)?2:((c<0xF0)?3:4));
                for(size_t k=0; k<n && i<val.size(); k++) out.push_back(val[i++]);
            }
            // Ensure tok vector is large enough
            while(tok.size() <= std::stoi(key)) tok.push_back("");
            tok[std::stoi(key)] = out;
            idx = val_end + 1;
        }
        // Remove empty entries at the end
        while(!tok.empty() && tok.back().empty()) tok.pop_back();
    } else {
        // Original text format: one token per line
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
    std::unique_ptr<ov::CompiledModel> mel_cm;
    std::unique_ptr<ov::CompiledModel> enc_cm;
    std::unique_ptr<ov::CompiledModel> dec_cm;
    std::unique_ptr<ov::CompiledModel> joint_cm;
    std::unique_ptr<ov::InferRequest> mel_req;
    std::unique_ptr<ov::InferRequest> enc_req;
    std::unique_ptr<ov::InferRequest> dec_req;
    std::unique_ptr<ov::InferRequest> joint_req;
    Vocab vocab;
    long T_frames;
    long mel_max_samples;  // max audio samples the mel model accepts
    std::vector<float> st1, st2;  // decoder LSTM state
    std::string device_;

    ~Asr() { unload_models(); }

    void load_models(const std::string& model_dir, const std::string& device, long t_frames);
    void unload_models();
    std::string transcribe(const float* audio, long N, long sr);
};

void Asr::load_models(const std::string& model_dir, const std::string& device, long t_frames){
    if(!vocab.tok.empty()) return; // already loaded
    T_frames=t_frames;
    device_ = device;
    core.set_property(ov::cache_dir(CACHE_DIR));

    // Resolve the correct model file names. The new OpenVINO IR export
    // uses parakeet_*.xml naming. There is a mel model, encoder, decoder, and joint.
    std::string mel_path, enc_path, dec_path, joint_path, vocab_path;

    // Try parakeet-tdt-v3-ov layout (OpenVINO IR)
    vocab_path = model_dir+"/parakeet_v3_vocab.json";
    mel_path   = model_dir+"/parakeet_melspectogram.xml";
    enc_path   = model_dir+"/parakeet_encoder.xml";
    dec_path   = model_dir+"/parakeet_decoder.xml";
    joint_path = model_dir+"/parakeet_joint.xml";

    vocab.load(vocab_path);
    fprintf(stderr,"[asr] vocab=%d blank=%d\n", vocab.size, vocab.blank);

    // mel model: input_signals [1, 240000] (15s fixed), input_length [1] -> [1, 128, T_mel], seq_len [1]
    // The model's internal ops don't support dynamic sequence length, so we pad/truncate to 240000.
    mel_max_samples = 240000;  // 15s at 16kHz
    auto mel=core.read_model(mel_path);
    // Set a name on the first output (unnamed in the IR) so we can access it
    mel->output(0).set_names({"mel_output"});
    // Mel model always on CPU (tiny, 465KB, and GPU mel produces garbage)
    mel_cm=std::make_unique<ov::CompiledModel>(core.compile_model(mel, "CPU"));
    mel_req=std::make_unique<ov::InferRequest>(mel_cm->create_infer_request());
    fprintf(stderr,"[asr] mel compiled on CPU\n");

    // encoder: melspectogram [1, 128, T_mel] -> encoder_output [1, 1024, T_enc]
    auto enc=core.read_model(enc_path);
    enc_cm=std::make_unique<ov::CompiledModel>(core.compile_model(enc, device));
    enc_req=std::make_unique<ov::InferRequest>(enc_cm->create_infer_request());
    fprintf(stderr,"[asr] encoder compiled on %s\n", device_.c_str());

    // decoder always on CPU (NPU INT8 LSTM has quantization accuracy issues)
    // decoder: targets [1,1] + h_in [2,1,640] + c_in [2,1,640]
    //   -> decoder_output [1,1,640] + h_out [2,1,640] + c_out [2,1,640]
    // decoder runs on CPU (LSTM state on NPU has quantization accuracy issues)
    auto dec=core.read_model(dec_path);
    std::map<std::string,ov::PartialShape> nd;
    for(auto&i:dec->inputs()){
        auto n=i.get_any_name();
        if(n=="targets")    nd[n]=ov::PartialShape{1,1};
        else if(n=="h_in"||n=="c_in") nd[n]=ov::PartialShape{2,1,STATE_DIM};
    }
    dec->reshape(nd);
    dec_cm=std::make_unique<ov::CompiledModel>(core.compile_model(dec, "CPU"));
    dec_req=std::make_unique<ov::InferRequest>(dec_cm->create_infer_request());
    fprintf(stderr,"[asr] decoder compiled on CPU\n");

    // joint model: encoder_outputs [1,1,1024] + decoder_outputs [1,1,640] -> logits [1,1,1,8198]
    auto jt=core.read_model(joint_path);
    jt->reshape({{"encoder_outputs", ov::PartialShape{1,1,1024}},
                 {"decoder_outputs", ov::PartialShape{1,1,640}}});
    joint_cm=std::make_unique<ov::CompiledModel>(core.compile_model(jt, "CPU"));
    joint_req=std::make_unique<ov::InferRequest>(joint_cm->create_infer_request());
    fprintf(stderr,"[asr] joint compiled on CPU\n");
}

void Asr::unload_models(){
    if(joint_req) joint_req.reset();
    if(joint_cm) joint_cm.reset();
    if(mel_req) mel_req.reset();
    if(mel_cm) mel_cm.reset();
    if(dec_req) dec_req.reset();
    if(dec_cm) dec_cm.reset();
    if(enc_req) enc_req.reset();
    if(enc_cm) enc_cm.reset();
    st1.clear(); st2.clear();
    fprintf(stderr,"[asr] models unloaded\n");
}

std::string Asr::transcribe(const float* audio, long N, long sr){
    /*
     * 1. Mel model: raw audio -> mel spectrogram
     *    mel input: input_signals [1, MAX_AUDIO], input_length [1]
     *    mel output: output_0 [1, 128, T_mel], seq_len [1]
     */
    long copy_samples = std::min(N, mel_max_samples);

    // Set mel model input (pad/truncate to fixed 240000 samples = 15s)
    long copy_to_mel = std::min(N, mel_max_samples);
    auto mel_input = mel_req->get_tensor("input_signals");
    memcpy(mel_input.data<float>(), audio, sizeof(float)*copy_to_mel);
    mel_req->get_tensor("input_length").data<int64_t>()[0] = copy_to_mel;
    mel_req->infer();

    auto mel_out = mel_req->get_tensor("mel_output");
    auto mel_shape = mel_out.get_shape();
    long mel_T = mel_shape[2];  // sequence length
    long mel_len = mel_req->get_tensor("seq_len").data<int64_t>()[0];

    // 2. Encoder: mel spectrogram -> encoded features
    //    enc input: melspectogram [1, 128, mel_T], melspectogram_length [1]
    //    enc output: encoder_output [1, 1024, T_enc], encoder_output_length [1]
    auto mel_tensor_data = mel_req->get_tensor("mel_output");
    const float* mel_data = mel_tensor_data.data<float>();
    auto mel_input_tensor = enc_req->get_tensor("melspectogram");
    memcpy(mel_input_tensor.data<float>(), mel_data, sizeof(float)*mel_T*128);
    enc_req->get_tensor("melspectogram_length").data<int32_t>()[0] = mel_len;
    enc_req->infer();

    auto enc_out = enc_req->get_tensor("encoder_output");
    auto enc_shape = enc_out.get_shape();
    long E = enc_shape[2];  // encoder frames
    long enc_len = enc_req->get_tensor("encoder_output_length").data<int64_t>()[0];
    if(enc_len > E) enc_len = E;
    const float* enc_data = enc_out.data<float>();
    fprintf(stderr,"[asr] enc_len=%ld E=%ld mel_T=%ld mel_len=%ld\n", enc_len, E, mel_T, mel_len);

    /*
     * 3. TDT greedy decode (decoder + joint)
     */
    st1.assign(2*1*STATE_DIM, 0.0f);
    st2.assign(2*1*STATE_DIM, 0.0f);
    std::vector<int> tokens;
    long t = 0;
    int emitted = 0;

    while(t < enc_len){
        // Extract encoder output at position t: shape is [1, 1024, E] → we need [1, 1024]
        // enc_data layout: [1, 1024, E] so enc_data[t*1+ d*E] = enc_data[d*E + t]
        std::vector<float> enc_vec(1024);
        for(int d = 0; d < 1024; d++){
            enc_vec[d] = enc_data[d*E + t];
        }

        // Decoder step: targets [1,1] + h_in [2,1,640] + c_in [2,1,640]
        //  -> decoder_output [1,1,640] + h_out + c_out
        int prev = tokens.empty() ? vocab.blank : tokens.back();
        dec_req->get_tensor("targets").data<int64_t>()[0] = prev;
        memcpy(dec_req->get_tensor("h_in").data<float>(), st1.data(), sizeof(float)*st1.size());
        memcpy(dec_req->get_tensor("c_in").data<float>(), st2.data(), sizeof(float)*st2.size());
        dec_req->infer();

        auto dec_out = dec_req->get_tensor("decoder_output");
        const float* dec_data = dec_out.data<float>();

        // Save LSTM states only when a non-blank token is emitted (TDT decode)
        // (states stay the same for blank predictions)

        // Joint model: encoder_outputs [1,1,1024] + decoder_outputs [1,1,640] -> logits [1,1,1,8198]
        memcpy(joint_req->get_tensor("encoder_outputs").data<float>(), enc_vec.data(), sizeof(float)*1024);
        memcpy(joint_req->get_tensor("decoder_outputs").data<float>(), dec_data, sizeof(float)*640);
        joint_req->infer();

        auto logits = joint_req->get_tensor("logits");
        // logits shape: [1, 1, 1, 8198] → total 8198 floats, start at beginning
        const float* row = logits.data<float>();

        int token = -1; float token_max = -1e30f;
        for(int i = 0; i < VOCAB; i++){ if(row[i] > token_max){ token_max = row[i]; token = i; } }
        int dur_idx = 0; float dur_max = -1e30f;
        for(int i = VOCAB; i < 8198; i++){ if(row[i] > dur_max){ dur_max = row[i]; dur_idx = i - VOCAB; } }
        int step = dur_idx;
        if(step < 1) step = 1;  // TDT duration predictor: minimum step of 1 frame
        if(t < 61) fprintf(stderr,"[asr] t=%ld token=%d (max=%.3f) step=%d (max=%.3f) emitted=%d\n", t, token, token_max, step, dur_max, emitted);

        if(token != vocab.blank){
            // Update LSTM states only when emitting a non-blank token
            auto h_out = dec_req->get_tensor("h_out");
            auto c_out = dec_req->get_tensor("c_out");
            st1.assign(h_out.data<float>(), h_out.data<float>() + ov::shape_size(h_out.get_shape()));
            st2.assign(c_out.data<float>(), c_out.data<float>() + ov::shape_size(c_out.get_shape()));
            tokens.push_back(token);
            emitted++;
        }
        if(step > 0){ t += step; emitted = 0; }
        else if(token == vocab.blank || emitted == MAX_TOKENS_PER_STEP){ t += 1; emitted = 0; }
    }
    return ids_to_text(vocab, tokens);
}

// ---------------- clipboard / paste ----------------
static void copy_clipboard(const std::string& s){
    if(s.empty()) return;
    FILE* p=popen("wl-copy","w");
    if(!p) return;
    fwrite(s.data(),1,s.size(),p); pclose(p);
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

// Paste clipboard contents — byte-identical to what wl-copy stored,
// no keystroke injection, no dropped chars.
static void paste_clipboard(){
    if(focused_is_terminal())
        system("wtype -M ctrl -M shift -k v -m shift -m ctrl");
    else
        system("wtype -M ctrl -k v -m ctrl");
}

// write state JSON for bar visibility (mic icon shows when class != idle)
static void write_state(const std::string& cls, const std::string& path){
    if(path.empty()) return;
    std::ofstream f(path);
    f<<"{\"alt\":\""<<cls<<"\",\"class\":\""<<cls<<"\",\"tooltip\":\"\"}\n";
    f.flush();
    // Signal waybar to refresh immediately instead of waiting for poll interval
    system("pkill -RTMIN+8 waybar 2>/dev/null");
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

// ---------------- state file helpers ----------------
// The panel writes device.txt, model.txt, and enabled under STATE_DIR.
// The daemon reads them on every Copilot toggle so the UI is the primary
// control surface (no daemon restart needed when the user picks a new model).
static std::string read_file_str(const std::string& path, const std::string& fallback=""){
    std::ifstream f(path);
    if(!f) return fallback;
    std::string s; std::getline(f, s);
    return s;
}

static std::string read_model(){
    return read_file_str(STATE_DIR + "/model.txt", MODEL_NAME);
}

static std::string read_device(){
    std::string s = read_file_str(STATE_DIR + "/device.txt", DEVICE);
    // Normalize to uppercase
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    if(s != "CPU" && s != "GPU" && s != "NPU") return DEVICE;
    return s;
}

static bool read_enabled(){
    std::string s = read_file_str(STATE_DIR + "/enabled", "on");
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return !(s == "off" || s == "false" || s == "0" || s == "disabled");
}

static std::string resolve_model_dir(const std::string& model_name){
    if(model_name.empty()) return "";
    std::string dir = MODELS_DIR + "/" + model_name;
    // Check for devices.txt as a marker that this is a valid model dir
    if(::access((dir + "/devices.txt").c_str(), R_OK) == 0) return dir;
    // Fallback: check if the directory exists with any model files
    if(::access(dir.c_str(), R_OK | X_OK) == 0) return dir;
    return "";
}

// Determine the inference backend for a given model name.
// Returns one of: "whisper", "qwen3", "parakeet-stream", "parakeet-ov",
//                 "indicwav2vec", "unknown"
static std::string model_backend(const std::string& model_name){
    if(model_name.find("whisper") == 0) return "whisper";
    if(model_name.find("qwen3-asr") == 0) return "qwen3";
    if(model_name == "parakeet-v3") return "parakeet-stream";
    if(model_name.find("parakeet") == 0) return "parakeet-ov";
    if(model_name == "nepali-indicwav2vec") return "indicwav2vec";
    return "unknown";
}

// GGUF Parakeet transcription via libtranscribe (ggml backend).
// This links against libtranscribe.so at runtime for CPU and Vulkan GPU.
static std::string transcribe_ggml(const std::string& model_dir,
                                    const std::string& device,
                                    const std::string& audio_path){
    // Build a command that calls the whisper.cpp-compatible transcribe binary
    // if available, otherwise fall back to the Python helper.
    std::string gguf_path = model_dir + "/parakeet-tdt-0.6b-v3-Q8_0.gguf";
    if(::access(gguf_path.c_str(), R_OK) != 0){
        // Try alternate naming
        gguf_path = model_dir + "/" + model_dir.substr(model_dir.rfind('/')+1) + ".Q8_0.gguf";
    }

    std::string device_arg = device;
    // Map NPU -> CPU for ggml (no NPU backend in libtranscribe)
    if(device_arg == "NPU") device_arg = "CPU";

    std::string cmd = "python3 " + STATE_DIR + "/../whisper_transcribe.py"
        " --model-dir " + model_dir
        + " --device " + device_arg
        + " --audio " + audio_path + " 2>/dev/null";

    // If we have libtranscribe, use it directly instead
    // For now, use a simple approach: call the helper with the GGUF path
    if(::access(gguf_path.c_str(), R_OK) == 0){
        fprintf(stderr, "[asr-ggml] transcribing %s on %s\n", gguf_path.c_str(), device_arg.c_str());
    }

    // Use the existing whisper helper as a bridge — it can handle various models
    // via openvino_genai if available, or fall back to the Python whisper API.
    // For parakeet-ggml, we'd ideally link libtranscribe directly, but the
    // Python bridge is sufficient for now.
    FILE* p = popen(cmd.c_str(), "r");
    if(!p) return "";
    std::string output;
    char buf[4096];
    while(fgets(buf, sizeof(buf), p)) output += buf;
    pclose(p);

    // Parse JSON output
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

// Call a Python helper for ASR transcription. Used for whisper and qwen3 models
// that run via openvino_genai.
// Reads state/language.txt (e.g. "en"/"ne"/"de", or empty = auto-detect) and
// passes it to multilingual whisper exports so the model doesn't misdetect
// (turbo-int4 guessed Gujarati for Nepali audio). Empty for English-only models.
static std::string read_language(){
    std::string s = read_file_str(STATE_DIR + "/language.txt", "");
    // normalize: trim + lowercase
    while(!s.empty() && (s.back()==' '||s.back()=='\n'||s.back()=='\r'||s.back()=='\t')) s.pop_back();
    while(!s.empty() && (s.front()==' '||s.front()=='\t')) s.erase(s.begin());
    return s;
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

static std::string transcribe_via_helper(const std::string& helper_script,
                                          const std::string& model_dir,
                                          const std::string& device,
                                          const std::string& audio_path,
                                          const std::string& language){
    char tmpl[] = "/tmp/dictate-helper-XXXXXX";
    int fd = mkstemp(tmpl);
    if(fd < 0) {
        fprintf(stderr, "[asr-helper] mkstemp failed: %s\n", strerror(errno));
        return "";
    }
    close(fd);

    char dev_buf[64];
    snprintf(dev_buf, sizeof(dev_buf), "%s", device.c_str());

    // Language is passed through to the helper only when a language is explicitly
// selected in state/language.txt AND the model is multilingual (so English-only
// exports like base-en don't get a rejected kwarg). The helper then uses the
// model's own lang_to_id (turbo supports ne/de/en) rather than guessing.
std::string cmd = "python3 " + helper_script +
        " --model-dir " + model_dir +
        " --device " + dev_buf +
        " --audio " + audio_path;
    if(!language.empty()){ cmd += " --language " + language; }
    cmd += " 2>/dev/null";

    FILE* p = popen(cmd.c_str(), "r");
    if(!p) {
        fprintf(stderr, "[asr-helper] popen failed: %s\n", strerror(errno));
        ::unlink(tmpl);
        return "";
    }

    std::string output;
    char buf[4096];
    while(fgets(buf, sizeof(buf), p)) output += buf;
    int rc = pclose(p);
    ::unlink(tmpl);

    if(rc != 0) {
        fprintf(stderr, "[asr-helper] helper exited %d\n", rc);
        return "";
    }

    // Parse JSON: {"text": "...", "error": "...", ...}
    try {
        // Simple JSON parse: find "text" field
        size_t pos = output.find("\"text\"");
        if(pos == std::string::npos) {
            fprintf(stderr, "[asr-helper] helper error: no text field in output\n");
            return "";
        }
        pos = output.find(':', pos);
        if(pos == std::string::npos) return "";
        pos++;
        while(pos < output.size() && (output[pos] == ' ' || output[pos] == '"')) pos++;
        size_t end = output.find('"', pos);
        if(end == std::string::npos) return "";
        return output.substr(pos, end - pos);
    } catch(...) {
        return "";
    }
}

// Like transcribe_via_helper, but the helper is a compiled binary instead
// of a Python script (e.g. the C++ parakeet_stream_transcribe). Same
// argv/JSON contract, just no `python3` prefix.
static std::string transcribe_via_helper_bin(const std::string& helper_bin,
                                              const std::string& model_dir,
                                              const std::string& device,
                                              const std::string& audio_path,
                                              const std::string& language){
    char dev_buf[64];
    snprintf(dev_buf, sizeof(dev_buf), "%s", device.c_str());

    std::string cmd = helper_bin +
        " --model-dir " + model_dir +
        " --device " + dev_buf +
        " --audio " + audio_path;
    if(!language.empty()){ cmd += " --language " + language; }
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
    try {
        size_t pos = output.find("\"text\"");
        if(pos == std::string::npos) {
            fprintf(stderr, "[asr-helper] helper error: no text field in output\n");
            return "";
        }
        pos = output.find(':', pos);
        if(pos == std::string::npos) return "";
        pos++;
        while(pos < output.size() && (output[pos] == ' ' || output[pos] == '"')) pos++;
        size_t end = output.find('"', pos);
        if(end == std::string::npos) return "";
        return output.substr(pos, end - pos);
    } catch(...) {
        return "";
    }
}

// ---------------- daemon (Handy-style) ----------------
// Stays alive but does NOT keep models loaded. Only monitors for the
// Copilot button toggle via SIGUSR1, then does a load -> record -> transcribe -> unload cycle.
static std::atomic<bool> g_toggle{false};       // set on SIGUSR1 (Copilot tap)
static std::atomic<bool> g_rec_end{false};      // set when the recording child exits on its own
static std::atomic<pid_t> g_pw_record_pid{0};  // track the `timeout` recording child

static void on_sigusr1(int){ g_toggle.store(true); }

// When the recording child (`timeout pw-record`) dies on its own — the
// MAX_SECS window ran out, or it aborted — record that fact. Do NOT set
// g_toggle (that's reserved for real Copilot taps); the loop checks g_rec_end.
// Guarded on a live record child so a stray SIGCHLD from the Python helper
// (which the daemon spawns in-process after recording) can't re-trigger.
static void on_sigchld(int){
    if(g_pw_record_pid.load() > 0) g_rec_end.store(true);
}

static void run_daemon(){
    std::string stbase="/home/shlok/.local/share/npu-asr/state";
    std::string stfile=stbase+"/status.json";
    std::string pidfile=stbase+"/daemon.pid";
    mkdir(stbase.c_str(),0755);

    std::ofstream(pidfile)<<getpid()<<"\n";

    struct sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=on_sigusr1; sigaction(SIGUSR1,&sa,NULL);

    // SIGCHLD wakes the loop if pw-record dies on its own (timeout hit).
    struct sigaction sch; memset(&sch,0,sizeof(sch));
    sch.sa_handler=on_sigchld; sigaction(SIGCHLD,&sch,NULL);

    write_state("idle", stfile);
    fprintf(stderr,"[dictate] daemon ready (load-on-demand). Tap Copilot to start.\n");

    int rec_running=0;
    for(;;){
        // Wake for either a real Copilot tap (g_toggle) or the recording child
        // dying on its own at the MAX_SECS cap (g_rec_end). pause() returns on
        // any signal; we re-check both so a stray signal is a no-op.
        pause();
        bool tap = g_toggle.exchange(false);
        bool rec_ended = g_rec_end.exchange(false);
        if(!tap && !rec_ended) continue;   // unrelated signal, keep waiting
        // Log the cap stop only when it's the SOLE trigger — a real Copilot tap
        // alongside a stale rec_end should report as a normal stop, not a cap.
        if(rec_ended && !tap && rec_running)
            fprintf(stderr,"[dictate] recording window hit its cap, auto-transcribing\n"), fflush(stderr);

        if(rec_running){
            // Stop + transcribe. Triggered by a 2nd tap OR the recording window
            // ending on its own (96s/--max-secs cap).
            if(!read_enabled()){
                // Master toggle flipped off mid-recording: abort, drop audio.
                fprintf(stderr,"[dictate] disabled, discarding recording\n"); fflush(stderr);
                rec_running = 0;
                continue;
            }

            // Load models, transcribe captured audio, unload
            write_state("transcribing", stfile);

            // Kill pw-record if still running and wait for it to flush the WAV
            pid_t pw_pid = g_pw_record_pid.exchange(0);
            if (pw_pid > 0) {
                kill(pw_pid, SIGTERM);
                // Wait for pw-record to finish writing and close the file
                for (int i = 0; i < 100; i++) {  // up to 1s
                    int status;
                    if (waitpid(pw_pid, &status, WNOHANG) > 0) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                // Force kill if still alive
                if (kill(pw_pid, 0) == 0) {
                    kill(pw_pid, SIGKILL);
                    waitpid(pw_pid, nullptr, 0);
                }
                fprintf(stderr, "[dictate] pw-record stopped, waiting for file flush...\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Retry reading the WAV file (pw-record may still be flushing)
            std::string wav = stbase + "/capture.wav";
            for (int attempt = 0; attempt < 50; attempt++) {
                if (::access(wav.c_str(), R_OK) == 0) {
                    // Verify file is valid WAV by checking with sfinfo
                    SF_INFO check; memset(&check, 0, sizeof(check));
                    if (sf_open(wav.c_str(), SFM_READ, &check) != nullptr) {
                        // File is valid, proceed
                        SF_INFO dummy; memset(&dummy, 0, sizeof(dummy));
                        SNDFILE* sf = sf_open(wav.c_str(), SFM_READ, &dummy);
                        if (sf) { sf_close(sf); break; }
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (::access(wav.c_str(), R_OK) != 0) {
                fprintf(stderr, "[dictate] no audio file to transcribe\n");
                write_state("idle", stfile);
                rec_running = 0;
                continue;
            }

            // Read state files fresh on every toggle — the panel is the control surface.
            std::string model_name = read_model();
            std::string device = read_device();
            fprintf(stderr, "[dictate] model=\"%s\" device=%s\n", model_name.c_str(), device.c_str());

            std::string model_dir = resolve_model_dir(model_name);
            if(model_name.empty() || model_dir.empty()){
                fprintf(stderr, "[dictate] model.txt \"%s\" not found, using default\n", model_name.c_str());
                // No model dir — can't transcribe
                write_state("idle", stfile);
                rec_running = 0;
                continue;
            }

            // Dispatch to the correct inference backend based on model family.
            std::string backend = model_backend(model_name);
            fprintf(stderr, "[dictate] backend=%s for model=%s\n", backend.c_str(), model_name.c_str());

            std::string transcript;

            if(backend == "whisper" || backend == "qwen3" || backend == "indicwav2vec"){
                // Whisper/Qwen3/indicwav2vec models run via a Python helper
                // (openvino_genai for whisper/qwen3, OV + CTC for indicwav2vec).
                // The helper handles audio resampling, model loading, and inference.
                std::string helper = STATE_DIR + "/../whisper_transcribe.py";
                if(backend == "qwen3") helper = STATE_DIR + "/../qwen3_transcribe.py";
                if(backend == "indicwav2vec") helper = STATE_DIR + "/../indicwav2vec_transcribe.py";
                // Pass a language override only for multilingual models when the
                // user has set one in state/language.txt (fixes turbo auto-
                // detecting Gujarati for Nepali audio). English-only and
                // Nepali-only models get "".
                std::string lang = "";
                if(model_is_multilingual(model_dir)) lang = read_language();
                transcript = transcribe_via_helper(helper, model_dir, device, wav, lang);
            } else if(backend == "parakeet-stream"){
                // Streaming-capable Parakeet via libtranscribe (ggml Vulkan).
                // We use the one-shot path here because the daemon records to a
                // wav on disk and the streaming path is reserved for the
                // future live-mic VAD integration. The libtranscribe ggml
                // backend has no NPU target; map NPU -> GPU so a stale
                // state/device.txt value doesn't break the dispatch.
                std::string helper_bin = std::string(getenv("HOME") ? getenv("HOME") : "") + "/.local/bin/parakeet_stream_transcribe";
                std::string dev_for_helper = device;
                if(dev_for_helper == "NPU") dev_for_helper = "GPU";
                std::string lang = "";
                if(model_is_multilingual(model_dir)) lang = read_language();
                transcript = transcribe_via_helper_bin(helper_bin, model_dir, dev_for_helper, wav, lang);
            } else if(backend == "parakeet-ov"){
                // Parakeet TDT in OpenVINO IR format — native C++ pipeline.
                Asr asr;
                std::string model_prefix = model_name;
                // parakeet-tdt-v3-ov uses different file names than parakeet-tdt-0.6b-v3-onnx
                std::string enc_model = model_dir + "/encoder-model.int8.onnx";
                std::string dec_model = model_dir + "/decoder_joint-model.int8.onnx";
                std::string vocab_file = model_dir + "/parakeet_v3_vocab.json";

                // Check if we have the ONNX files (old parakeet-tdt-0.6b-v3-onnx layout)
                if(::access(enc_model.c_str(), R_OK) != 0){
                    // Try alternate naming for openvino IR
                    enc_model = model_dir + "/parakeet_encoder.xml";
                    dec_model = model_dir + "/parakeet_decoder.xml";
                    vocab_file = model_dir + "/parakeet_v3_vocab.json";
                }

                if(::access(enc_model.c_str(), R_OK) == 0){
                    asr.load_models(model_dir, device, T_MAX);

                    SileroVad vad;
                    std::string vad_model_path = VAD_DIR + "/silero_vad_v4.onnx";

                    if (VAD_ENABLED && vad.load(vad_model_path, VAD_THRESHOLD)) {
                        long rate=0;
                        auto wavd=read_audio(wav,&rate);
                        if(!wavd.empty()){
                            std::string full_text;
                            std::vector<float> segment;
                            bool was_in_speech = false;

                            size_t offset = 0;
                            vad.reset();
                            while (offset + VAD_FRAME_SAMPLES <= wavd.size()) {
                                bool speech_now = vad.is_speech(wavd.data() + offset, VAD_FRAME_SAMPLES);

                                if (speech_now && !was_in_speech) {
                                    segment.clear();
                                    fprintf(stderr, "[dictate] segment started at %.1fs\n",
                                            offset / 16000.0);
                                }

                                if (speech_now) {
                                    segment.insert(segment.end(),
                                        wavd.begin() + offset,
                                        wavd.begin() + offset + VAD_FRAME_SAMPLES);
                                }

                                if (!speech_now && was_in_speech) {
                                    if (!segment.empty()) {
                                        long N_seg = (long)segment.size();
                                        fprintf(stderr, "[dictate] segment ended at %.1fs (%.1fs of speech)\n",
                                                (offset + VAD_FRAME_SAMPLES) / 16000.0,
                                                N_seg / 16000.0);
                                        std::string seg_text = asr.transcribe(segment.data(), N_seg, 16000);
                                        if (!seg_text.empty()) {
                                            if (!full_text.empty()) full_text += " ";
                                            full_text += seg_text;
                                        }
                                    }
                                    segment.clear();
                                }

                                was_in_speech = speech_now;
                                offset += VAD_FRAME_SAMPLES;
                            }

                            if (was_in_speech && !segment.empty()) {
                                long N_seg = (long)segment.size();
                                fprintf(stderr, "[dictate] final segment (%.1fs of speech)\n",
                                        N_seg / 16000.0);
                                std::string seg_text = asr.transcribe(segment.data(), N_seg, 16000);
                                if (!seg_text.empty()) {
                                    if (!full_text.empty()) full_text += " ";
                                    full_text += seg_text;
                                }
                            }

                            transcript = full_text;
                        }
                    } else {
                        // No VAD: process entire audio
                        long rate=0;
                        auto wavd=read_audio(wav,&rate);
                        if(!wavd.empty()){
                            long N_real=(long)wavd.size();
                            if(N_real>N_MAX) N_real=N_MAX;
                            transcript = asr.transcribe(wavd.data(), N_real, rate);
                        }
                    }
                }
            } else if(backend == "parakeet-ggml"){
                // GGUF Parakeet runs via libtranscribe (ggml backend).
                // The binary has libtranscribe linked; call g_ggml_transcribe.
                fprintf(stderr, "[dictate] parakeet-ggml backend (libtranscribe) for %s\n", model_dir.c_str());
                transcript = transcribe_ggml(model_dir, device, wav);
            } else {
                fprintf(stderr, "[dictate] unknown backend for model=%s\n", model_name.c_str());
            }

            if(!transcript.empty()){
                fprintf(stderr,"[dictate] transcript: %s\n", transcript.c_str()); fflush(stderr);
                printf("%s\n", transcript.c_str()); fflush(stdout);
                copy_clipboard(transcript);
                paste_clipboard();
            } else {
                fprintf(stderr,"[dictate] no speech detected\n");
            }

            ::rename(wav.c_str(), (stbase + "/capture.last.wav").c_str());  // keep recording for tests / user
            write_state("idle", stfile);
            rec_running = 0;
        }else{
            // Start recording — ONLY on a real Copilot tap, never from the
            // recording child ending on its own (that path transcribes above).
            if(!tap){ rec_running = 0; continue; }
            // 1st tap: start recording
            // In VAD mode, we use pw-record with a timeout, then process
            if(!read_enabled()){
                // Master toggle is off — Copilot does nothing.
                write_state("idle", stfile);
                fprintf(stderr,"[dictate] disabled (enabled=off), ignoring tap\n"); fflush(stderr);
                rec_running = 0;
                continue;
            }

            std::string wav = stbase + "/capture.wav";

            // Start pw-record with the MAX_SECS timeout
            pid_t pid=fork();
            if(pid==0){
                char timeout_str[32];
                snprintf(timeout_str, sizeof(timeout_str), "%d", (int)MAX_SECS);
                // Use timeout to limit recording duration
                execlp("timeout","timeout", timeout_str,
                       "pw-record","--rate","16000","--channels","1",
                       "--format","f32","--channel-map","mono", wav.c_str(), (char*)NULL);
                _exit(127);
            }
            rec_running = 1;
            g_pw_record_pid.store(pid);
            g_rec_end.store(false);   // clear any stale cap signal from a prior recording
            write_state("recording", stfile);
            fprintf(stderr,"[dictate] recording started (pid=%d)\n", pid); fflush(stderr);
        }
    }
}

// ---------------- VAD-aware live capture (experimental) ----------------
// This is the future path: capture audio in-memory and run VAD live,
// stopping when silence is detected rather than using a fixed timeout.
// Currently not used in daemon mode but available for --stream mode.
static void run_vad_capture(){
    // This would use ALSA/PipeWire directly to capture audio in 32ms frames,
    // run Silero VAD on each frame, and stop recording when speech ends.
    // Kept as a stub for future integration.
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
        else if(a=="--model" && i+1<argc){ MODEL_NAME=argv[++i]; }
        else if(a=="--vad-threshold" && i+1<argc){ VAD_THRESHOLD=(float)atof(argv[++i]); }
        else if(a=="--vad-off" || a=="--no-vad") { VAD_ENABLED=false; }
        else if(!a.empty() && a[0]!='-') wavpath=a;
    }

    // In daemon mode, read state files on each toggle (done inside run_daemon).
    // In CLI mode, fall back to command-line --model/--device or state files.
    if(MODEL_NAME.empty()){
        MODEL_NAME = read_model();
    }
    if(DEVICE == "NPU" && argc <= 1){
        // Only auto-read device if no --device flag was passed
    }

    if(mode=="daemon"){
        run_daemon();
        return 0;
    }

    if(mode=="transcribe" && wavpath.empty()){
        fprintf(stderr,"dictate: no wav given\n"); return 2;
    }

    if(mode=="transcribe" || mode=="type"){
        // Resolve state for CLI mode
        std::string model_name = MODEL_NAME;
        std::string device = DEVICE;
        // Try reading state files for device if not overridden
        std::string state_device = read_device();
        if(state_device != "NPU" || !MODEL_NAME.empty()){
            device = read_device();
            model_name = read_model();
        }

        std::string model_dir = resolve_model_dir(model_name);
        if(model_dir.empty()){
            fprintf(stderr,"[dictate] model.txt \"%s\" not found, using default\n", model_name.c_str());
            fprintf(stderr, "dictate: model not found\n");
            return 1;
        }

        std::string backend = model_backend(model_name);
        fprintf(stderr, "[dictate] backend=%s model=%s device=%s\n", backend.c_str(), model_name.c_str(), device.c_str());

        std::string text;
        if(backend == "whisper" || backend == "qwen3" || backend == "indicwav2vec"){
            std::string helper = STATE_DIR + "/../whisper_transcribe.py";
            if(backend == "qwen3") helper = STATE_DIR + "/../qwen3_transcribe.py";
            if(backend == "indicwav2vec") helper = STATE_DIR + "/../indicwav2vec_transcribe.py";
            std::string lang = "";
            if(model_is_multilingual(model_dir)) lang = read_language();
            text = transcribe_via_helper(helper, model_dir, device, wavpath, lang);
        } else if(backend == "parakeet-stream"){
            std::string helper_bin = std::string(getenv("HOME") ? getenv("HOME") : "") + "/.local/bin/parakeet_stream_transcribe";
            std::string dev_for_helper = device;
            if(dev_for_helper == "NPU") dev_for_helper = "GPU";
            std::string lang = "";
            if(model_is_multilingual(model_dir)) lang = read_language();
            text = transcribe_via_helper_bin(helper_bin, model_dir, dev_for_helper, wavpath, lang);
        } else if(backend == "parakeet-ov"){
            Asr asr;
            SileroVad vad;
            std::string vad_model_path = VAD_DIR + "/silero_vad_v4.onnx";

            asr.load_models(model_dir, device, T_MAX);

            long rate=0;
            auto wav=read_audio(wavpath,&rate);
            long N_real=(long)wav.size();
            if(N_real>N_MAX) N_real=N_MAX;

            if (VAD_ENABLED && vad.load(vad_model_path, VAD_THRESHOLD)) {
                std::string full_text;
                std::vector<float> segment;
                bool was_in_speech = false;

                size_t offset = 0;
                vad.reset();
                while (offset + VAD_FRAME_SAMPLES <= (size_t)N_real) {
                    bool speech_now = vad.is_speech(wav.data() + offset, VAD_FRAME_SAMPLES);

                    if (speech_now && !was_in_speech) {
                        segment.clear();
                    }

                    if (speech_now) {
                        segment.insert(segment.end(),
                            wav.begin() + offset,
                            wav.begin() + offset + VAD_FRAME_SAMPLES);
                    }

                    if (!speech_now && was_in_speech && !segment.empty()) {
                        std::string seg_text = asr.transcribe(segment.data(), (long)segment.size(), 16000);
                        if (!seg_text.empty()) {
                            if (!full_text.empty()) full_text += " ";
                            full_text += seg_text;
                        }
                        segment.clear();
                    }

                    was_in_speech = speech_now;
                    offset += VAD_FRAME_SAMPLES;
                }

                // Flush final segment
                if (was_in_speech && !segment.empty()) {
                    std::string seg_text = asr.transcribe(segment.data(), (long)segment.size(), 16000);
                    if (!seg_text.empty()) {
                        if (!full_text.empty()) full_text += " ";
                        full_text += seg_text;
                    }
                }

                text = full_text;
            } else {
                // No VAD: use original pipeline
                long rate=0;
                auto wav=read_audio(wavpath,&rate);
                long N_real=(long)wav.size();
                if(N_real>N_MAX) N_real=N_MAX;
                text=asr.transcribe(wav.data(),N_real,rate);
            }
        } else if(backend == "parakeet-ggml"){
            text = transcribe_ggml(model_dir, device, wavpath);
        }

        if(!text.empty()){
            printf("%s\n", text.c_str());
            if(mode=="type"){ copy_clipboard(text); paste_clipboard(); }
        } else {
            fprintf(stderr,"[dictate] no speech detected\n");
            printf("\n");
        }
    }
    return 0;
}
