# Contributing to WSE

Bug reports, fixes, documentation, examples, and new operators, adapters, evaluators, or strategies are welcome. Everyone who takes part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Report a bug

[Open an issue](https://github.com/xuguang-ai/weight-space-engineering/issues) with:

- what you ran: the command or code, and the WSE version (`python -c "import wse; print(wse.__version__)"`);
- the model and the commit that `Model.from_pretrained` recorded, which the first line of `model.inspect()` shows;
- your platform: the operating system, Python, PyTorch, and, for agent runs, `codex --version`;
- the full error output.

Leave out keys, endpoints, and data you cannot share.

## Report a security problem privately

A way for an agent's code to read the test data, your keys, or files outside its run, or to change WSE's records, is a security problem. Do not open a public issue for it: report it through [GitHub's private vulnerability reporting](https://github.com/xuguang-ai/weight-space-engineering/security/advisories/new), which only you and the maintainer can see.

## Set up

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/xuguang-ai/weight-space-engineering.git
cd weight-space-engineering
uv run --all-extras pytest -q
```

`uv run --all-extras` creates the environment from `uv.lock`, with every optional extra and the test tools. The suite runs offline on tiny models built from configuration classes: it needs no key, downloads no model, and took about 11 minutes on an M4 Pro. On a machine without a Codex CLI signed by OpenAI, or without macOS's `sandbox-exec`, the tests that need them are skipped.

## Make a change

- **A failing test first.** For a bug, add a test that fails on the current code, then fix the code.
- **The full suite before a pull request,** not only the tests near your change.
- **Edits stay exact.** Every edit is an exact patch that replays to identical weights; a change to patches, exports, or replay needs a test that replays.
- **Extensions go through the registry.** [docs/extensions.md](docs/extensions.md) gives the contracts of operators, adapters, evaluators, and strategies.
- **Examples run as written.** Code in the README and the docs must run as shown, and any output shown must come from a run of that code. Measurements go in [docs/results.md](docs/results.md) with the date, the commit, the hardware, and the settings, including runs in which nothing improved.
- **No secrets or large files.** Never commit a `.env`, keys, endpoints, model weights, or run outputs; `.gitignore` already leaves out `.env`, `*.safetensors`, `*.sqlite`, and `runs/` and `artifacts/` at the repository root.

## Live agent runs

An agent run calls your model provider with your key and costs money; the offline suite never does. To run one, follow the README's [Quick start](README.md#quick-start). In a source checkout, you can keep the key in a `.env` file copied from [.env.example](.env.example), which git ignores, and load it for one command with `uv run --env-file .env`.

## Pull requests

Open a pull request against `main`. Say what changed and why, and how you tested it: the suite's result, and for a live run, the agent's model, the Codex version, and the date. Keep one change per pull request.

## License

WSE is licensed under the [Apache License 2.0](LICENSE). Unless you state otherwise, a contribution you submit for inclusion is licensed under the same terms (Section 5 of the license).
