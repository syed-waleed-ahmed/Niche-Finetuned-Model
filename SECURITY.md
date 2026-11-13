# Security policy

## Supported versions

| Version | Supported | Notes |
| --- | --- | --- |
| 2.x | Yes | The `main` branch and the container images built from it (`ghcr.io`, tags `latest` and `sha-<commit>`). |
| 1.x | No | The single-origin service tagged `v1.0.0`. Upgrade to 2.x. |
| Before 1.0.0 | No | The original training scripts. |

Fixes are made on `main` and shipped by the Release workflow. There are no backports
to older versions.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability
reporting, which is enabled for this repository:

1. Open the repository's **Security** tab, or go directly to
   [the new advisory form](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/security/advisories/new).
2. Choose **Report a vulnerability**.
3. Fill in the advisory form.

Do not open a public issue, pull request or discussion that describes a vulnerability.
If you cannot use GitHub's form, open a public issue titled "Security contact request"
that contains no technical details. The maintainer will then set up a private channel.

A useful report includes:

- the affected component (for example the API service, the edge Worker, the web UI,
  the container image, or a workflow) and the version or commit;
- steps to reproduce, or a proof of concept;
- the impact you expect (what an attacker gains);
- any fix or mitigation you suggest.

Never include real API keys, tokens, or other credentials in a report, and redact
them from any logs you attach.

### Testing guidelines

- Test against a local deployment (`fastapi-assistant serve`, or
  `docker compose up --build`) wherever possible.
- Do not run load tests, denial-of-service attempts, or automated scanners against
  the public deployment. It runs on capped free-tier capacity, and doing so harms
  other users.
- Do not access, modify, or delete data that is not yours, and stop as soon as you
  have shown that a problem exists.

## Response targets

This is a single-maintainer project. The following are goals, not guarantees:

| Step | Target |
| --- | --- |
| Acknowledge the report | within 5 business days |
| Initial assessment and severity | within 10 business days |
| Fix or mitigation, critical or high severity | within 30 days of confirmation |
| Fix or mitigation, medium or low severity | within 90 days of confirmation |

Fixes are disclosed in a GitHub security advisory once a fixed version is deployed.
Reporters are credited in the advisory unless they ask not to be.

## Scope

In scope:

- the Python service in `src/fastapi_assistant/`: HTTP API, authentication and API
  keys, admission control, request limits, storage, retrieval, and model loading
  (including the revision and SHA-256 pinning in `model-release.json`);
- the Cloudflare Worker in `deploy/cloudflare-worker/` (key checks, rate limits,
  answer cache, origin failover, origin lock);
- the web UI in `web/`;
- the container image built from `Dockerfile`, and the images published from this
  repository;
- the GitHub Actions workflows in `.github/workflows/` and the deployment templates in
  `deploy/` and `scripts/`;
- the dependency locks (`requirements/serve.txt`, `package-lock.json` files).

Out of scope:

- incorrect, low-quality, or made-up model answers. The model only produces text and
  has no tools or data access, so a wrong answer is a quality bug: please open a bug
  report. See [MODEL_CARD.md](MODEL_CARD.md) for known limitations;
- prompt injection that only changes the text of the model's own answer, without
  crossing a security boundary;
- `fastapi-assistant eval --execute-code`, which runs model-generated code by design
  and is documented for disposable environments only (report it if that code can run
  without the flag);
- vulnerabilities in third-party platforms (Cloudflare, Google Cloud, Hugging Face,
  Neon, Upstash, Kaggle, GitHub). Report those to the vendor;
- vulnerabilities in upstream dependencies with no demonstrated impact on this
  project. Report those upstream; tell us if you can show an impact here;
- volumetric denial of service, social engineering, and physical attacks;
- scanner output without a demonstrated impact.

## Accepted risks

These are known, reviewed, and accepted. Reports that only restate them will be
closed with a pointer to this section.

### PYSEC-2026-2447: diskcache 5.6.3

- **What:** unsafe deserialisation of cached values in `diskcache` 5.6.3. No fixed
  release exists.
- **Why it is present:** `llama-cpp-python` depends on `diskcache` for its optional
  `LlamaDiskCache`, so the serving lock (`requirements/serve.txt`) includes it.
- **Why it is accepted:** the service never constructs a `LlamaDiskCache` (or any
  llama.cpp prompt cache), so the vulnerable code path is not reachable.
- **How it is tracked:** the `pip-audit` job in `.github/workflows/security.yml`
  ignores this one advisory ID (`--ignore-vuln PYSEC-2026-2447`). Every other
  advisory fails the job, and the job also runs weekly. The exception will be removed
  when a fixed `diskcache` release exists, or if the service ever starts using a disk
  cache.

### Other accepted findings

The initial audit, [docs/audit/INITIAL_AUDIT.md](docs/audit/INITIAL_AUDIT.md),
records further accepted findings with their reasons, including:

- **F-15:** two CodeQL alerts that are intended behaviour: `keys create` prints a new
  key once to the operator's terminal, and API keys (256-bit random values) are
  hashed with SHA-256;
- **F-16:** `/health` and `/ready` are public;
- **F-18:** CORS allows any origin, because authentication never uses cookies.

## Security controls and further reading

- [docs/security/SECURITY_AUDIT.md](docs/security/SECURITY_AUDIT.md): security audit
  of the hardened service.
- [docs/audit/INITIAL_AUDIT.md](docs/audit/INITIAL_AUDIT.md): baseline audit, with
  evidence for every finding.
- Automated checks on every pull request: CodeQL, gitleaks (full history), Trivy on
  the container image (fails on fixable HIGH and CRITICAL CVEs), `pip-audit` on the
  serving lock, `npm audit` for the Worker and the web UI, and GitHub dependency
  review.
