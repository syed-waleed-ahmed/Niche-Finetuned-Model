# Getting help

| You want to | Where |
| --- | --- |
| Report a bug | [Open a bug report](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/issues/new/choose); the template asks for the version, the steps and the response you got |
| Suggest a feature | [Open a feature request](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/issues/new/choose) |
| Report a security vulnerability | Privately, as described in [SECURITY.md](SECURITY.md). Never in a public issue |
| Ask how something works | Start with the [documentation index](README.md#documentation) and the [runbook](docs/operations/RUNBOOK.md); if the answer is not there, ask in [Discussions](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/discussions) and say what you tried |
| Share an idea or show what you built | [Discussions](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/discussions) |
| See what is planned | [ROADMAP.md](ROADMAP.md) |
| Contribute a change | Read [CONTRIBUTING.md](CONTRIBUTING.md), pick an issue labelled [good first issue](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/labels/good%20first%20issue) if you are new, then open a pull request |

## Before opening an issue

- Search the [existing issues](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/issues),
  including closed ones.
- Include the `X-Request-ID` of the failing request if you have one. Every response
  carries it, and it links the edge and origin logs.
- For a wrong answer from the model, include the question and the answer. The served
  model is small (1.1B parameters), so wrong answers are expected. They are tracked as
  evaluation data, not as bugs, unless the service itself misbehaves.
- Never paste API keys, tokens or other secrets.

This is a community project maintained in spare time. There is no guaranteed response
time, but every well-described issue is read.
