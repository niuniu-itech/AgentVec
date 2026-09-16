# Configuration

Copy the variable names from `.env.example` into your shell or a local environment
manager. The package does not load `.env` implicitly. Keep secrets outside source
control. Manual generation and CPU tests require neither an API key nor SSH.

For SiliconFlow, set `SILICONFLOW_API_KEY` and `SILICONFLOW_MODEL`. `SF_KEY` remains
an alias for existing scripts. Use a provider model ID available to your account;
historical aliases in the artifact are not a live availability list. The client
never changes models after an error. Cache identity includes endpoint, requested
model, prompt, temperature, token limit and thinking mode. Cache records preserve
the returned model name and token usage when the provider supplies them.

Use `AGENTVEC_CACHE_DIR` for an external cache directory. A reused response is a
cached experiment, not a fresh independent LLM sample. Do not compare cached and
uncached call time as if both were inference time.

SSH configuration has three roles:

| Role | Variables | Use |
| --- | --- | --- |
| `server` | `AGENTVEC_SERVER_HOST`, `_USER`, `_KEY` or `_PASSWORD`, `_PORT` | Cross-compile and QEMU, using `_GCC` and `_QEMU` |
| `board` | `AGENTVEC_BOARD_HOST`, `_USER`, `_KEY` or `_PASSWORD`, `_PORT` | Native RISC-V compile/test or execution |
| `ascend` | `AGENTVEC_ASCEND_HOST`, `_USER`, `_KEY` or `_PASSWORD`, `_PORT` | CANN compilation and Ascend310P1 validation |

An x86/GPU server can serve as the cross-build host; these checks do not use its
GPU. Use the `ascend` command and role for an Ascend 310P host. See
[Ascend setup](ascend.md) for toolkit, CMake and private environment setup options.

Remote runs use a unique subdirectory of `AGENTVEC_REMOTE_ROOT` (default
`/tmp/agentvec`). They do not overwrite previous runs. `--gcc` and `--qemu` override
tool selection for a single invocation. The driver captures stdout, stderr, source
hashes and verification results in the explicit local `--output` directory.
