// parakeet_stream_transcribe_rt.cpp — REAL-TIME streaming helper for live mic.
//
// Unlike parakeet_stream_transcribe (which reads a WAV file and emits one final
// JSON), this binary reads RAW float32 PCM (16kHz mono) from stdin in real time
// and emits incremental JSON lines on each fed chunk:
//   {"text": "...", "is_final": false}
//   {"text": "...", "is_final": true}
//
// The daemon pipes pw-record's stdout directly to this binary's stdin.
//
// Args:
//   --model-dir PATH    Directory containing the GGUF
//   --device NAME       CPU | GPU  (GPU maps to "vulkan" for libtranscribe)
//   --chunk-ms N        Feed cadence in ms (default 320 = 5.12s at 16kHz... no, 320ms)
//   --chunk-ctx-ms N    Left context window (default 1040)
//   --left-ms N         Left context for chunked attention (default 5600)
//   --right-ms N        Right context for chunked attention (default 1040)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <signal.h>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <cerrno>
#include <sys/select.h>

#include "transcribe.h"
#include "transcribe/parakeet.h"

#include <atomic>
namespace {

struct Args {
    std::string model_dir;
    std::string gguf_path;
    std::string device = "CPU";
    int chunk_ms = 320;
    int left_ms = 5600;
    int chunk_ctx_ms = 1040;
    int right_ms = 1040;
    // Silence auto-stop (streaming only): stop after this many
    // seconds of continuous sub-threshold audio. 0 = disabled.
    // Armed only after the first FED speech chunk, so a slow mic start can
    // never end the session. Measured over FED AUDIO seconds, not
    // wall-clock, so a slow decode under GPU load can't false-trigger.
    double silence_stop_s = 0.0;
    double silence_rms = 0.01;
};

void usage(){
    std::fprintf(stderr,
        "usage: parakeet_stream_transcribe_rt --model-dir DIR --device CPU|GPU\n"
        "       [--chunk-ms N] [--left-ms N] [--chunk-ctx-ms N] [--right-ms N]\n"
        "       [--silence-stop-s SECS] [--silence-rms RMS]\n"
        "Reads raw float32 PCM (16kHz mono) from stdin, emits JSON lines on stdout.\n");
    std::exit(2);
}

bool parse_int(const char* s, int& out){
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if(end == s || *end != '\0') return false;
    out = (int)v;
    return true;
}

bool parse_double(const char* s, double& out){
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if(end == s || *end != '\0') return false;
    out = v;
    return true;
}

Args parse_args(int argc, char** argv){
    Args a;
    for(int i=1; i<argc; ++i){
        std::string k = argv[i];
        auto next = [&](const char* what){
            if(i+1 >= argc){ std::fprintf(stderr, "missing arg for %s\n", what); std::exit(2); }
            return std::string(argv[++i]);
        };
        if(k == "--model-dir") a.model_dir = next("--model-dir");
        else if(k == "--device") a.device = next("--device");
        else if(k == "--chunk-ms"){ if(!parse_int(argv[++i], a.chunk_ms)) usage(); }
        else if(k == "--left-ms"){  if(!parse_int(argv[++i], a.left_ms)) usage(); }
        else if(k == "--chunk-ctx-ms"){ if(!parse_int(argv[++i], a.chunk_ctx_ms)) usage(); }
        else if(k == "--right-ms"){ if(!parse_int(argv[++i], a.right_ms)) usage(); }
        else if(k == "--silence-stop-s"){ if(!parse_double(argv[++i], a.silence_stop_s)) usage(); }
        else if(k == "--silence-rms"){ if(!parse_double(argv[++i], a.silence_rms)) usage(); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(); }
    }
    if(a.model_dir.empty()) usage();

    // Resolve gguf path
    a.gguf_path = a.model_dir + "/parakeet-unified-en-0.6b-Q8_0.gguf";
    if(::access(a.gguf_path.c_str(), R_OK) != 0){
        std::string cmd = "ls -1 " + a.model_dir + "/*.gguf 2>/dev/null | head -1";
        FILE* p = popen(cmd.c_str(), "r");
        if(p){
            char buf[1024] = {0};
            if(std::fgets(buf, sizeof(buf), p)){
                size_t n = std::strlen(buf);
                while(n && (buf[n-1]=='\n'||buf[n-1]=='\r')) buf[--n] = 0;
                if(n) a.gguf_path = buf;
            }
            pclose(p);
        }
    }
    if(a.gguf_path.empty() || ::access(a.gguf_path.c_str(), R_OK) != 0){
        std::fprintf(stderr, "[parakeet-rt] no gguf found under %s\n", a.model_dir.c_str());
        std::exit(1);
    }
    return a;
}

void emit_inc_json(const std::string& text, bool is_final, double tx_s = 0.0){
    std::string esc;
    esc.reserve(text.size()+8);
    for(char c : text){
        switch(c){
            case '"':  esc += "\\\""; break;
            case '\\': esc += "\\\\"; break;
            case '\n': esc += "\\n";  break;
            case '\r': esc += "\\r";  break;
            case '\t': esc += "\\t";  break;
            default:
                if((unsigned char)c < 0x20) esc += '?';
                else esc += c;
        }
    }
    std::printf("{\"text\":\"%s\",\"is_final\":%s,\"transcribe_seconds\":%.3f}\n",
                esc.c_str(), is_final ? "true" : "false", tx_s);
    std::fflush(stdout);
}

} // namespace

static std::atomic<bool> g_running{true};
static void on_sigint(int){ g_running.store(false); }

int main(int argc, char** argv){
    Args a;
    try { a = parse_args(argc, argv); }
    catch(const std::exception& e){ emit_inc_json("", true, 0); return 1; }

    // Map device -> libtranscribe backend name
    std::string backend = "cpu";
    if(a.device == "GPU" || a.device == "gpu") backend = "vulkan";
    else if(a.device == "CPU" || a.device == "cpu") backend = "cpu";
    else { std::fprintf(stderr, "[parakeet-rt] device %s not supported\n", a.device.c_str()); return 1; }

    // TAP-INSTANT SPOOL: drain stdin on a side thread while the main thread
    // runs backend init / model load / session init / stream begin. Audio
    // spoken from the tap instant (before the model is ready) is buffered in
    // RAM instead of sitting in the 1MB pipe with no reader, so it can never
    // backpressure pw-record no matter how long the load takes. Only the
    // spool thread touches `early` until join; only main touches it after.
    // All writers emit multiples of 4B and all reads request multiples of
    // 4B, so a short tail is not expected; dropping it matches the
    // steady-state read path below.
    int fl0 = fcntl(STDIN_FILENO, F_GETFL, 0);
    if(fl0 >= 0) fcntl(STDIN_FILENO, F_SETFL, fl0 | O_NONBLOCK);
    std::vector<float> early;
    // Unbounded session (streaming has no record cap): grow on demand.
    // Seed covers 60s; vector growth handles longer takes (~7.7MB/min).
    early.reserve(16000 * 60);
    std::atomic<bool> early_eof{false};
    std::atomic<bool> spool_stop{false};
    std::thread spool([&]{
        float b[4096];
        while(!spool_stop.load() && g_running.load()){
            fd_set rfds; FD_ZERO(&rfds); FD_SET(STDIN_FILENO, &rfds);
            struct timeval tv; tv.tv_sec = 0; tv.tv_usec = 100 * 1000;
            int r = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
            if(r < 0) break;
            if(r > 0 && FD_ISSET(STDIN_FILENO, &rfds)){
                ssize_t n = read(STDIN_FILENO, b, sizeof(b));
                if(n == 0){ early_eof.store(true); break; }
                else if(n > 0){
                    size_t nf = (size_t)n / sizeof(float);
                    early.insert(early.end(), b, b + nf);
                } else if(errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR){
                    break;
                }
            }
        }
    });
    auto stop_spool = [&]{ spool_stop.store(true); if(spool.joinable()) spool.join(); };

    // Init backends
    {
        std::string dir = a.model_dir + "/../ggml";
        transcribe_status s = transcribe_init_backends(dir.c_str());
        if(s != TRANSCRIBE_OK){
            s = transcribe_init_backends_default();
            if(s != TRANSCRIBE_OK){
                std::fprintf(stderr, "[parakeet-rt] transcribe_init_backends failed\n");
                stop_spool();
                return 1;
            }
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    transcribe_model* model = nullptr;
    transcribe_model_load_params lp;
    transcribe_model_load_params_init(&lp);
    transcribe_status s = transcribe_model_load_file(a.gguf_path.c_str(), &lp, &model);
    if(s != TRANSCRIBE_OK || !model){
        std::fprintf(stderr, "[parakeet-rt] model load failed\n");
        stop_spool();
        return 1;
    }
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[parakeet-rt] model loaded in %.3fs\n", load_s);

    // Build session
    transcribe_session* session = nullptr;
    transcribe_session_params sps;
    transcribe_session_params_init(&sps);
    s = transcribe_session_init(model, &sps, &session);
    if(s != TRANSCRIBE_OK || !session){
        std::fprintf(stderr, "[parakeet-rt] session init failed\n");
        stop_spool();
        transcribe_model_free(model);
        return 1;
    }

    // Set up streaming
    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);

    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    sp.commit_policy = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;

    transcribe_parakeet_buffered_stream_ext ext;
    transcribe_parakeet_buffered_stream_ext_init(&ext);
    ext.left_ms  = a.left_ms;
    ext.chunk_ms = a.chunk_ctx_ms;
    ext.right_ms = a.right_ms;
    sp.family = &ext.ext;

    s = transcribe_stream_begin(session, &rp, &sp);
    if(s != TRANSCRIBE_OK){
        std::fprintf(stderr, "[parakeet-rt] stream begin failed\n");
        stop_spool();
        transcribe_session_free(session);
        transcribe_model_free(model);
        return 1;
    }
    // Load done: stop the spool. Its buffer becomes the head of `pending`
    // below, so tap-instant audio is fed first, in order, then live mic.
    spool_stop.store(true);
    if(spool.joinable()) spool.join();

    // Read PCM from stdin in chunks
    const int sr = 16000;
    const int chunk_samples = (a.chunk_ms * sr) / 1000;
    std::vector<float> pcm_chunk(chunk_samples);

    std::string committed;   // last emitted display string (dedup only)
    std::string authoritative; // full_text snapshot: the model's current
                               // truth over everything fed so far. committed_
                               // text is append-only/best-effort and can get
                               // stuck (commits freeze; finalize only appends
                               // compatible suffix), so it must never be the
                               // final transcript — full_text is.
    auto t1 = std::chrono::steady_clock::now();

    // Non-blocking stdin: buffer whatever the mic already delivered and
    // only feed when a full chunk is ready. A blocking fread here would
    // sit on a partial chunk while unread bytes pile up behind it,
    // delaying the first hypothesis (start-loss fix).
    int fl = fcntl(STDIN_FILENO, F_GETFL, 0);
    if(fl >= 0) fcntl(STDIN_FILENO, F_SETFL, fl | O_NONBLOCK); // re-assert; already set pre-spool

    std::vector<float> pending;
    pending.reserve(chunk_samples * 2 + early.size());
    pending.insert(pending.end(), early.begin(), early.end());
    early.clear(); early.shrink_to_fit();
    // A short take may have EOF'd while the model was still loading; the
    // existing finalize path below handles that.
    bool eof_seen = early_eof.load();

    // Silence auto-stop state: ~20 lines of doubles, no deps, runs on the
    // audio already fed to the model — near-zero cost (one RMS over the
    // 320ms feed chunk). Armed only after the FIRST speech chunk has been
    // fed, so a slow/quiet mic start can never end the session early.
    // silence_stop_s==0 disables (other models / prod paths never pass it).
    bool speech_seen = false;
    double fed_audio_s = 0.0;
    double silence_s = 0.0;
    const double chunk_s = (double)chunk_samples / sr;
    bool silence_stop = false;

    while(g_running.load()){
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO, &rfds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100 * 1000;
        int sel = eof_seen ? 1 : select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
        if(sel < 0) goto finalize;
        if(sel > 0 && !eof_seen && FD_ISSET(STDIN_FILENO, &rfds)){
            float buf[4096];
            ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
            if(n == 0){
                eof_seen = true;
            } else if(n > 0){
                size_t nf = (size_t)n / sizeof(float);
                pending.insert(pending.end(), buf, buf + nf);
            } else if(errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR){
                goto finalize;
            }
        }
        // EOF (file-redirect input drained): feed any leftover partial
        // audio — it may hold the last word — then finalize.
        if(eof_seen && pending.size() < pcm_chunk.size()){
            if(!pending.empty()){
                s = transcribe_stream_feed(session, pending.data(), (int)pending.size(), nullptr);
                if(s != TRANSCRIBE_OK){
                    std::fprintf(stderr, "[parakeet-rt] stream feed failed, finalzing\n");
                    goto finalize;
                }
                pending.clear();
            }
            goto finalize;
        }
        // Feed every full chunk available (usually one); leftover partial
        // audio stays buffered for the next pass or finalize. Feeding
        // promptly keeps the first hypothesis latency low (start-loss fix).
        while(pending.size() >= pcm_chunk.size()){
            std::copy(pending.begin(), pending.begin() + pcm_chunk.size(), pcm_chunk.begin());
            pending.erase(pending.begin(), pending.begin() + pcm_chunk.size());
            // Silence tracker: RMS over this exact fed chunk. NaN-safe:
            // the daemon tee already zeroes NaN/Inf upstream, but guard
            // anyway (NaN comparison is false, so sum!=sum catches it).
            if(a.silence_stop_s > 0.0){
                double sumsq = 0.0;
                for(int k = 0; k < chunk_samples; k++){
                    float v = pcm_chunk[k];
                    sumsq += (double)v * (double)v;
                }
                double rms = (sumsq == sumsq) ? std::sqrt(sumsq / chunk_samples) : 1.0;
                if(rms >= a.silence_rms){
                    speech_seen = true;
                    silence_s = 0.0;
                } else if(speech_seen){
                    silence_s += chunk_s;
                    if(silence_s >= a.silence_stop_s){
                        std::fprintf(stderr, "[parakeet-rt] silence auto-stop: %.1fs quiet\n", silence_s);
                        silence_stop = true;
                    }
                }
            }
            s = transcribe_stream_feed(session, pcm_chunk.data(), (int)pcm_chunk.size(), nullptr);
            if(s != TRANSCRIBE_OK){
                std::fprintf(stderr, "[parakeet-rt] stream feed failed, finalzing\n");
                goto finalize;
            }
            fed_audio_s += chunk_s;
            if(silence_stop) goto finalize;
        }

        // Get incremental text: display = committed (stable) + tentative (volatile).
        // Committed alone stays empty for long stretches on parakeet; showing
        // the tentative hypothesis gives the live ~1s-delayed streaming view.
        // ALSO snapshot full_text every pass: it is the authoritative raw
        // hypothesis and the finalize source of truth.
        transcribe_stream_text out;
        transcribe_stream_text_init(&out);
        if(transcribe_stream_get_text(session, &out) == TRANSCRIBE_OK){
            if(out.full_text){
                authoritative.assign(out.full_text, out.full_text_bytes);
            }
            std::string current;
            if(out.committed_text){
                current.assign(out.committed_text, out.committed_text_bytes);
            }
            if(out.tentative_text){
                current.append(out.tentative_text, out.tentative_text_bytes);
            }
            // Only emit if the display text changed
            if(current != committed){
                committed = current;
                double tx_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
                emit_inc_json(committed, false, tx_s);
            }
        }
    }

finalize:
    {
        double tx_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        s = transcribe_stream_finalize(session, nullptr);
        transcribe_stream_text out;
        transcribe_stream_text_init(&out);
        if(transcribe_stream_get_text(session, &out) == TRANSCRIBE_OK){
            // full_text is authoritative: the raw final hypothesis over ALL
            // fed audio. Prefer it over committed_text, which is append-only
            // and best-effort — it can freeze mid-utterance (head-loss) and
            // finalize only appends compatible suffix bytes, never repairs.
            std::string fin;
            if(out.full_text){
                fin.assign(out.full_text, out.full_text_bytes);
            }
            if(fin.empty() && out.committed_text){
                fin.assign(out.committed_text, out.committed_text_bytes);
            }
            if(fin.empty()){
                fin = authoritative;
            }
            committed = fin;
        } else if(!authoritative.empty()){
            committed = authoritative;
        }
        emit_inc_json(committed, true, tx_s);
    }

    transcribe_session_free(session);
    transcribe_model_free(model);
    return 0;
}
