# Pilot container preparation

## Build from the repository root

Use a running Docker daemon in Linux-container mode:

```console
docker version
docker build -t tg-acac:pilot-v1 .
```

Intel cluster architecture is linux/amd64; on an ARM build host add
--platform linux/amd64. Building needs no GPU, but requires Docker Hub, Debian
repositories, PyPI, the PyTorch wheel index and several GiB of disk space.

Python 3.9.25 and PyTorch 2.1.0/CUDA 12.1 match the validated core environment.
The official Python tag/digest was checked against Docker Hub.
The legacy setup.py bundle is not installed. Build checks include pip check,
the full unit suite, runtime versions and CLI help; no training starts.
Transitive Python/apt versions are not fully locked. Use ONE built image digest
across all compared methods/seeds. Resolved Python versions are saved at
/usr/local/share/tg-acac/requirements-resolved.txt.

Source, tests, analysis and graphics are included. Data, checkpoints, papers,
Git metadata and credentials are excluded. Uncommitted SOURCE changes are
included: record/commit reviewed experiment source before publishing a release.

## Validate

```console
docker run --rm tg-acac:pilot-v1
docker run --rm tg-acac:pilot-v1 python -m unittest discover -s tests -q
docker run --rm --gpus all tg-acac:pilot-v1 python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Default CMD prints help. The GPU check requires NVIDIA-enabled Docker.
Driver 535.129.03 satisfies CUDA 12.1's requirement; runtime/device plugin setup
must still be verified. No build-time GPU test is claimed.

## Output and short smoke (prepared, not executed)

Working directory is /workspace. ACAC_DATA_DIR=/data; mount durable storage there.
Windows PowerShell, from the repository root, with an existing data directory:

```powershell
docker run --rm --mount "type=bind,source=${PWD}/data,target=/data" tg-acac:pilot-v1 python acac/scripts/acac_main.py --alg_name acac --env_name ovcA7 --exp_name container_smoke_cpu_v1 --seed 0 --device cpu --total_epi 16 --eval_freq 8 --eval_num_epi 2 --record_pilot_metrics true --profile_resources true
```

Use fresh experiment names on repeats. GPU smoke adds --gpus all, --device cuda
and a new experiment name. CPU smoke does not establish GPU access or learning
quality. Cluster calibration precedes any long pilot.

## Cluster values for later Job templates

- Image: harbor.n705.work/n705/fcc/tg-acac
- Namespace: n705
- imagePullSecrets: harbor-n705-work (exists; actual pull unverified)
- PVC: fcc, bound, 2Ti, ReadWriteMany, rook-cephfs; mount at /data
- Initial calibration proposal: 1 GPU, 8 CPU cores, 8Gi memory, n_env=8
- Servers 1-3: two A800 80GB each; 4-5: two RTX 4090 each
- Kubernetes node names/labels and current GPU availability remain to verify.

The UI shows DEFAULT container limits of 10 CPU cores/8Gi memory, not evidence
of hard maxima. Visible project quotas have no upper bound; resources may be busy.

## Harbor upload after validation (prepared, not executed)

```console
docker login harbor.n705.work
docker tag tg-acac:pilot-v1 harbor.n705.work/n705/fcc/tg-acac:pilot-v1
docker push harbor.n705.work/n705/fcc/tg-acac:pilot-v1
```

Build-host push credentials are separate from the cluster pull Secret.
Record the uploaded digest; reuse it for all comparison runs.

## Verification status

The agent shell cannot find docker. Actual image build, Linux tests, container
CPU/GPU smoke, upload and cluster deployment remain pending.
No image upload, cluster submission, long training or git push was performed.

Local Windows regression: all 45 tests passed under one OMP/MKL thread and
dummy SDL; git diff --check passed. Linux build remains pending.
