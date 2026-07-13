# Kangri TTS Webservice

A containerized Flask web service exposing the fine-tuned Kangri SpeechT5 model over
HTTP, so text or marked-up files can be sent to this machine to generate downloadable
wav files. Wraps the same code the CLI scripts use (`../src/infer.py`,
`../src/infer_file.py`) rather than reimplementing anything, so behavior stays in sync
with the CLI.

## Usage

```powershell
# Build + run with GPU, using the default project path (C:\vscode\buildCombinedDataset)
.\buildDocker.ps1 -Gpu

# Enable the optional compare mode and synthesize-file endpoint too
.\buildDocker.ps1 -Gpu `
  -WavDir "C:\Users\pete_\Dropbox\NTprogress\PahariAudio\KangriWordDownloads\FCBH\wavs" `
  -CharacterMappingDir "C:\My Paratext 9 Projects\xnr\shared\milestone-markers"

# CPU-only, custom port
.\buildDocker.ps1 -Port 8080
```

Then browse to http://localhost:8000/ (or your chosen `-Port`) for a simple test UI, or
http://localhost:8000/docs for interactive Swagger UI / the OpenAPI spec (see
[Generating a C#/.NET client](#generating-a-cnet-client) below), or call the API directly
(see below).

## What gets mounted, and why

Unlike the reference NLLB translator project (which bakes source code into the image
and mounts only the model), this project mounts the **entire buildCombinedDataset
project** (`src/`, `checkpoints/`, `data/`, `model_init*/`, `tokenizer*/`) read-only at
`/app/project` inside the container. `src/infer.py`'s own path logic
(`ROOT = Path(__file__).parent.parent`) then resolves correctly automatically, since it's
running from the mounted copy -- no path-rewriting needed. This also means editing
`src/infer.py` on the host and restarting the container picks up the change without
rebuilding the image.

Two more mounts are **optional**, each only needed for one feature:
- `-WavDir`: raw reference recordings, needed only for the `compare` option of
  `POST /api/v1/tts/synthesize/`. Without it, that option returns a clear error.
- `-CharacterMappingDir`: folder containing `characterMapping.json`, needed only for
  `POST /api/v1/tts/synthesize-file/`. Without it, that endpoint returns a clear error.

The server itself never writes into any of these mounts -- results are streamed back in
the HTTP response, not saved to disk on the host (unlike running the CLI scripts
directly, which write to `outputs/`, `ALL_CHARACTERS_ROOT`, etc).

## API Endpoints (on localhost:&lt;port&gt;)

### GET /api/v1/tts/health/
```json
{"status": "ok", "loaded_variants": ["b"]}
```

### GET /api/v1/tts/characters/?variant=a|b
Character list ranked by likely voice quality (same ranking as `infer.py --list-characters`):
```json
{
  "variant": "b",
  "characters": [
    {"rank": 1, "character": "Jesus", "trainMinutes": 1110.5, "trainUtterances": 9612, "valUtterances": 516},
    ...
  ]
}
```

### POST /api/v1/tts/synthesize/
JSON body:
```json
{"text": "...", "character": "Jesus", "variant": "b"}
```
Returns `audio/wav` directly.

Pass `"character": "all"` (equivalent to `infer.py --character` with no value) to get
every character's voice for that text instead, as an `application/zip` download
containing `01_Jesus.wav`, `02_Paul.wav`, ... and a `manifest.json`.

Pass `"compare": N` (with `"character"` set) to get `N` random held-out validation
examples for that character synthesized and zipped alongside their real recordings
(equivalent to `infer.py --compare N --character ...`) -- requires `-WavDir` to have been
mounted.

### POST /api/v1/tts/synthesize-file/
Async -- a whole file can take several minutes, too long for most HTTP clients to hold a
connection open for. Either multipart form-data with a `file` field (a spkrEmb-marked-up
text file, see `../src/infer_file.py`'s docstring for the format), or JSON
`{"text": "<file content>"}`. Optional form/JSON field `variant`. Requires
`-CharacterMappingDir` to have been mounted.

Returns immediately with **202 Accepted** and `{"jobId": "...", "status": "pending", "statusUrl": "..."}`.
The job runs in a background thread; poll:

### GET /api/v1/tts/jobs/&lt;jobId&gt;/
```json
{"jobId": "...", "status": "running"}
```
`status` is one of `pending`, `running`, `done`, `error` (with an `error` message field),
or `done` (with `synthCount` and a `downloadUrl` field). A reasonable poll interval is
~30s -- there's no push/webhook notification, only polling.

### GET /api/v1/tts/jobs/&lt;jobId&gt;/download/
Returns `audio/wav` once the job's `status` is `done`. `409` if it's not ready yet, `404`
if the job id is unknown (including after a server restart -- jobs are in-memory only,
not persisted).

## OpenAPI / Swagger

The full API is described in `openapi.yaml` (OpenAPI 3.0.3), served at
`GET /openapi.yaml`, with an interactive Swagger UI at `GET /docs`. Both are unauthenticated
regardless of `API_KEY` (they only describe the API; they don't call it).

### Generating a C#/.NET client

[NSwag](https://github.com/RicoSuter/NSwag) is the most straightforward way to get a
strongly-typed client for .NET Framework 4.8 (it generates `HttpClient`-based code, which
.NET 4.8 supports natively via the built-in `System.Net.Http` assembly):

1. Get NSwag: either **NSwagStudio** (Windows GUI, simplest for a one-off/manual
   regeneration) from the NSwag releases page, or the `dotnet-nswag` CLI tool
   (`dotnet tool install -g NSwag.ConsoleCore`) if you'd rather script/automate it.
2. Point it at `http://<this-machine's-address>:8000/openapi.yaml` as the input, choose
   "CSharpClient" generation, and pick a namespace/output file.
3. **Binary responses**: since every synthesize endpoint returns `audio/wav` or
   `application/zip` rather than JSON, NSwag generates each of those operations returning
   a `FileResponse` (wraps a `Stream` + headers + status code, and is `IDisposable`) --
   not a POCO. That's expected; write the `Stream` to disk or hand it to whatever plays/
   saves the audio on the .NET side.
4. **Polling for synthesize-file**: that endpoint is async (see above) -- call
   `SynthesizeFileAsync(...)`, take the returned `jobId`, then call
   `GetJobStatusAsync(jobId)` in a loop (e.g. every 30s via `System.Threading.Timer` or a
   simple `Task.Delay` loop) until `status` is `"done"` or `"error"`, then call
   `DownloadJobResultAsync(jobId)` to get the `FileResponse`.
5. **Auth**: if the server was started with `-ApiKey`, add an
   `Authorization: <key>` header to the generated client's `HttpClient` (NSwag-generated
   clients expose a constructor overload or partial method for this, depending on
   generation settings) -- it's checked for an exact string match, not a `Bearer` scheme.
6. Since this is a native `HttpClient` call (not a browser), CORS doesn't apply.

## Configuration

All via environment variables (see `settings.py`), set through `buildDocker.ps1`
parameters rather than edited by hand:
- `PORT`, `API_KEY` (require `Authorization: <key>` header if set), `DEFAULT_VARIANT` (`a`/`b`)
- `PROJECT_PATH`, `WAV_DIR`, `CHARACTER_MAPPING_DIR` -- container-side paths, set
  automatically by `buildDocker.ps1` based on the `-v` mounts it creates.

## Troubleshooting

- **"Project src directory not found"**: `-ProjectPath` doesn't point at a valid
  buildCombinedDataset checkout, or the volume mount failed.
- **"compare requires WAV_DIR..." / "synthesize-file requires CHARACTER_MAPPING_DIR..."**:
  rerun `buildDocker.ps1` with `-WavDir` / `-CharacterMappingDir` set.
- **Browser UI shows "Unexpected token '<' ... is not valid JSON"**: the server hit an
  unhandled exception and Flask returned its default HTML error page. Check
  `docker logs <container>` for the real traceback -- as of this writing the only known
  cause is the next item.
- **"RuntimeError: Failed to find C compiler" in `docker logs`**: recent `torch` routes
  some ops (e.g. `bmm` inside SpeechT5's attention) through a Triton-JIT-compiled kernel,
  which needs a C compiler in the container to build itself on first use. Fixed by
  installing `build-essential` in the Dockerfile (already included) -- if you see this,
  rebuild the image (`.\buildDocker.ps1`) rather than just restarting the container.
- **GPU not used**: rerun without `-Gpu`, or check `docker/health/` response and your
  NVIDIA Container Toolkit install (`docker run --gpus` requires it on the host).
- **Slow first request per variant**: each transcription variant's model/dataset loads
  lazily on first use and is cached in memory afterward -- the first `a` request and the
  first `b` request will each be slow; subsequent ones are fast.

## Local Development (No Docker)

```powershell
$env:PROJECT_PATH = "C:\vscode\buildCombinedDataset"
cd C:\vscode\buildCombinedDataset\docker
..\.venv\Scripts\python.exe server.py
```
