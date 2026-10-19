# MacBook validation

macOS compatibility is implemented for Apple Silicon and Intel on macOS 14+.
A physical MacBook has not yet been validated. A Linux/macOS CI matrix is prepared locally;
publishing GitHub Actions files currently requires adding `workflow` scope to the GitHub token.
The matrix tests installed-plugin acceptance without downloading models or qualifying GPU inference.

The foundation is a separate [repository](https://github.com/GeorgeQLe/agentic-mcp-foundation).
Local-vision pins its exact Git commit; a clean clone requires no sibling checkout or vendored files.

## Clean-clone acceptance

Install Git, Python 3.10+, uv, and FFmpeg. With an existing Homebrew installation:

```bash
brew install python uv ffmpeg
git clone https://github.com/GeorgeQLe/local-vision-mcp.git
cd local-vision-mcp
python3 scripts/macos_acceptance.py --live
```

The command checks locked dependencies, lint, tests, verified plugin installation and offline doctor,
then provisions latest Ollama and runs a real Qwen3.5 2B color probe on a dedicated endpoint. It does
not require an existing Ollama installation or administrator privileges for the managed runtime.
The upstream universal archive keeps its runner/Metal libraries beside the binary. Ollama selects
acceleration; this project does not install drivers or force the optional Omni backend onto MPS.

Without `--live`, the command verifies the Python/MCP lifecycle without downloading Ollama or models.
Allow several gigabytes and time for the first live download. An existing readable cache can be
reused with `--reuse-models "$HOME/.ollama/models"`; the original cache is not modified. Add
`--root "$HOME/Pictures"` to allow that media directory in the test installation. Test paths use a
new temporary installation by default; an explicit `--home` must not contain an installed payload.

The script prints the path to `macbook-acceptance.json`. A live run also retains `live-report.json`,
including the exact model, Ollama version, producer identity and probe receipt. It restores test
settings and stops its owned test Ollama afterward. A passing color probe establishes basic vision
operation, not OCR/video/audio quality or performance. Keep both reports with the tested Git commit.

## Enable and exercise the plugin

For a persistent installation with your actual media roots:

```bash
python3 scripts/install.py install --root "$HOME/Pictures"
python3 "$HOME/.local/share/local-vision-mcp/control/launcher.py" --doctor
```

Enable `~/.local/share/local-vision-mcp/plugin` through your MCP host's plugin flow and refresh the
connection. Check `local_media_status`; call `local_media_manage` with `action: "ensure"` and a
stable idempotency key. Poll the returned job, then submit a screenshot job from an allowed root.
Disconnect and reconnect the MCP client and recover the original job ID. Installed launchers do not
need the source checkout or a host plugin cache. If a GUI host cannot find Python or Homebrew's
FFmpeg, configure its executable/PATH to include your Python and `/opt/homebrew/bin` (Apple Silicon)
or `/usr/local/bin` (Intel), then reconnect.

An existing Ollama desktop/Homebrew service may be reused, but local-vision will refuse to stop or
update it because it does not own it. The dedicated acceptance endpoint avoids changing that service.

The archive/layout and OS requirement follow [Ollama's macOS documentation](https://docs.ollama.com/macos)
and [upstream Darwin packaging](https://github.com/ollama/ollama/blob/main/scripts/build_darwin.sh).
Process ownership uses Apple's [libproc process information](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info.h).
