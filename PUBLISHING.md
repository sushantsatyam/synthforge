# Publishing SynthForge

## Distribution Channels

| Channel | Audience | Command |
|---|---|---|
| **PyPI** | Public Python users | `pip install synthforge` |
| **TestPyPI** | Testing before real publish | `pip install -i https://test.pypi.org/simple synthforge` |
| **GitHub Packages** | Org-internal distribution | `pip install --index-url https://...` |
| **Private PyPI** | Enterprise (DevPI/Artifactory) | `pip install --index-url https://your-server/simple synthforge` |
| **Conda-forge** | Conda users | `conda install -c conda-forge synthforge` |
| **Docker** | Containerized deployments | `docker pull ghcr.io/yourname/synthforge` |

---

## 1. PyPI (Public)

### One-time setup

```bash
# 1. Create account at https://pypi.org/account/register/
# 2. Enable 2FA (required)
# 3. Create API token at https://pypi.org/manage/account/token/
# 4. Configure ~/.pypirc:
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
EOF
chmod 600 ~/.pypirc
```

### Publish

```bash
# Build
pip install build twine
python -m build

# Verify
twine check dist/*

# Upload to TestPyPI first (always test first!)
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple synthforge

# If all good, upload to real PyPI
twine upload dist/*
```

### Trusted Publishing (Recommended — no tokens needed)

Instead of API tokens, use GitHub OIDC (already configured in `.github/workflows/ci.yml`):

```
1. Go to https://pypi.org/manage/project/synthforge/settings/publishing/
2. Add a "GitHub" trusted publisher:
   - Owner: YOUR_GITHUB_USERNAME
   - Repository: synthforge
   - Workflow: ci.yml
   - Environment: pypi
3. Push a version tag: git tag v0.1.0 && git push --tags
4. GitHub Actions will auto-publish to PyPI
```

---

## 2. TestPyPI (Staging)

```bash
# Upload
twine upload --repository testpypi dist/*

# Install
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple synthforge
```

---

## 3. GitHub Packages

```bash
# Build
python -m build

# Upload using GitHub token
twine upload \
  --repository-url https://upload.pypi.org/legacy/ \
  --username __token__ \
  --password $GITHUB_TOKEN \
  dist/*
```

Or use the GitHub Actions workflow (already configured — triggers on `v*` tags).

---

## 4. Private PyPI Server (Enterprise)

### Option A: DevPI (self-hosted)

```bash
# Install devpi server
pip install devpi-server devpi-client

# Start server
devpi-server --start --init

# Create index
devpi use http://localhost:3141
devpi login root --password ''
devpi index -c root/private

# Upload
devpi upload dist/*

# Install from private server
pip install -i http://localhost:3141/root/private/+simple/ synthforge
```

### Option B: JFrog Artifactory

```bash
# Configure twine
twine upload \
  --repository-url https://your-company.jfrog.io/artifactory/api/pypi/python-local \
  --username YOUR_USERNAME \
  --password YOUR_TOKEN \
  dist/*

# Install
pip install \
  --index-url https://your-company.jfrog.io/artifactory/api/pypi/python-local/simple \
  synthforge
```

### Option C: AWS CodeArtifact

```bash
# Get auth token
TOKEN=$(aws codeartifact get-authorization-token \
  --domain your-domain \
  --query authorizationToken \
  --output text)

# Upload
twine upload \
  --repository-url https://your-domain-123456789.d.codeartifact.us-east-1.amazonaws.com/pypi/your-repo/ \
  --username aws \
  --password $TOKEN \
  dist/*

# Install
pip install \
  --index-url https://aws:${TOKEN}@your-domain-123456789.d.codeartifact.us-east-1.amazonaws.com/pypi/your-repo/simple/ \
  synthforge
```

### Option D: Google Artifact Registry

```bash
# Configure
pip install keyrings.google-artifactregistry-auth

# Upload
twine upload \
  --repository-url https://us-central1-python.pkg.dev/YOUR_PROJECT/YOUR_REPO/ \
  dist/*

# Install
pip install \
  --index-url https://us-central1-python.pkg.dev/YOUR_PROJECT/YOUR_REPO/simple/ \
  synthforge
```

---

## 5. Conda-Forge

Create `recipe/meta.yaml`:

```yaml
package:
  name: synthforge
  version: "0.1.0"

source:
  url: https://pypi.io/packages/source/s/synthforge/synthforge-0.1.0.tar.gz
  sha256: <SHA256_OF_TARBALL>

build:
  noarch: python
  script: python -m pip install . -vv
  entry_points:
    - synthforge = synthforge.cli:main

requirements:
  host:
    - python >=3.10
    - pip
    - setuptools >=68.0
  run:
    - python >=3.10
    - pandas >=2.0,<3.0
    - numpy >=1.24,<2.1
    - scikit-learn >=1.3
    - scipy >=1.11
    - pydantic >=2.0
    - tqdm >=4.65
    - faker >=20.0

test:
  imports:
    - synthforge
  commands:
    - synthforge --help

about:
  home: https://github.com/YOUR_USERNAME/synthforge
  license: LicenseRef-Proprietary
  summary: Synthetic data generation with LLM-augmented pipelines
```

Then submit to conda-forge: https://conda-forge.org/docs/maintainer/adding_pkgs/

---

## 6. Docker Image

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY dist/synthforge-*.whl .
RUN pip install synthforge-*.whl && rm *.whl

ENTRYPOINT ["synthforge"]
```

```bash
# Build
docker build -t ghcr.io/yourname/synthforge:0.1.0 .

# Push to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u yourname --password-stdin
docker push ghcr.io/yourname/synthforge:0.1.0

# Run
docker run -v $(pwd):/data ghcr.io/yourname/synthforge:0.1.0 \
  /data/input.csv -o /data/output.csv -n 100000
```

---

## Version Bumping & Release Workflow

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit
git add pyproject.toml CHANGELOG.md
git commit -m "Release v0.1.1"

# 4. Tag (triggers CI/CD → auto-publish to PyPI)
git tag v0.1.1
git push origin main --tags

# 5. GitHub Actions handles:
#    - Run tests on Python 3.10/3.11/3.12
#    - Lint with ruff
#    - Build sdist + wheel
#    - Publish to PyPI (trusted publishing)
#    - Create GitHub Release with artifacts
```

---

## Verification After Publishing

```bash
# Check PyPI listing
pip index versions synthforge

# Install fresh
pip install synthforge==0.1.0

# Verify
python -c "import synthforge; print(synthforge.__version__)"
synthforge --help
```
