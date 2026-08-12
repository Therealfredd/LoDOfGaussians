# LoD Trainer

A standalone Windows launcher for training Gaussian splats with
[A LoD of Gaussians](https://github.com/FelixWindisch/LoDOfGaussians) (SIGGRAPH 2026)
on your own COLMAP datasets.

Pick a folder, set your parameters, press Start. The app installs and builds
everything it needs the first time you run it.

## Running it

Double-click **`Launch LoD Trainer.bat`**.

The GUI itself only needs Python with tkinter (stdlib), so it starts before
anything else is installed and bootstraps the rest from there.

## Three tabs

**1. Setup** — shows what was detected on your machine (Python, Git, GPU, CUDA
toolkit, MSVC compiler) and installs the rest. Press *Install / Repair
everything*. It clones the repository, creates an isolated virtual environment
in `venv/`, installs a matching PyTorch build, compiles the CUDA extensions
(`simple-knn`, `gaussianhierarchy`, `fused-ssim`), and builds the
`GaussianHierarchyCreator` binary with CMake.

First run takes roughly 15–40 minutes; the CUDA compiles dominate. Everything
lands inside this folder — `repo/` and `venv/`. Nothing else on the system is
touched.

**2. Train** — pick the folder containing your COLMAP data. The app validates
the layout and tells you what is wrong before you waste hours on a failed run.
Choose a preset, then Start.

If the layout isn't what the trainer expects, **Prep** lights up. It handles the
three things that stop real-world COLMAP exports from loading:

- **images loose in the project root** rather than in `images/` — a common
  export shape where the photos and the model files all sit in one folder
- **the model in the wrong place** — in the project root, or in `sparse/`
  without the `0` subfolder. Files are moved into `sparse/0/` and their names
  normalised (`Cameras.txt` → `cameras.txt`), matched case-insensitively
- **cameras that are not `PINHOLE`.** The loader asserts `PINHOLE` exactly, but
  exports are frequently `FULL_OPENCV`, `OPENCV` or `SIMPLE_RADIAL` with all
  distortion coefficients set to zero — already undistorted, just labelled
  differently. Those are rewritten to `PINHOLE`, which is lossless; focal
  length, principal point and image size carry over untouched.

So a folder like this works straight away:

```
my-capture/
├─ 00001.png … 00036.png
├─ Cameras.txt
├─ Images.txt
└─ Points3D.txt
```

Prep lists exactly what it will change and asks first. Files are moved, not
copied, and the original camera file is kept beside the new one as
`.original`. Running it twice is a no-op. If the cameras carry *real*
distortion, it refuses and tells you to run `colmap image_undistorter` —
converting would silently corrupt the result.

**3. Parameters** — every knob that matters, grouped and explained. Values are
merged over the paper defaults and written to `repo/configs/lod_trainer.json`.

Three things there are worth knowing:

- **Max image width** controls training resolution (upstream's `-r`). `-1`
  downscales anything wider than 1600px to 1600; any value above 8 is the width
  in pixels. Careful: `1`, `2`, `4` and `8` mean *divide the original width by
  this* rather than "set the width" — an upstream quirk, not a typo.
- **Cache sizes are in GB**, not Gaussian counts. The field shows the count it
  works out to as you type. The conversion follows the model layout — 14 floats
  per Gaussian plus 3 channels of SH coefficients, all float32 — so it changes
  with SH degree: 92 bytes at degree 1, 236 at degree 3. Worth noting the paper
  default of 22M Gaussians is only **1.88 GB** at degree 1, so a 24 GB card has
  far more headroom than the default uses.
- **Presets are saveable.** *Save as...* stores everything currently set,
  including image width, under your own name; *Delete* removes it. Your presets
  sit alongside the built-in ones in the dropdown and persist in
  `settings.json`.

## Saving mid-run

**Save .ply now** on the Train tab lights up while training and writes
`snapshot_iteration_<N>.ply` into the output folder at the next iteration —
useful for checking progress without waiting for the run to finish, or for
salvaging something from a run you're about to stop.

It works by dropping a flag file that the trainer picks up on its next pass, so
the GUI never touches the model while it is being optimised. A snapshot that
fails is caught and logged rather than taking the run down with it — worth
knowing because upstream's `save_ply` hardcodes a 3×3 SH reshape and only
succeeds at **SH degree 1**.

Settings persist in `settings.json` between sessions.

## Expected dataset layout

```
<your folder>/
├─ sparse/0/
│  ├─ cameras.bin      (or .txt)
│  ├─ images.bin       (or .txt)
│  └─ points3D.bin     (or .txt)
├─ images/
├─ depths/             (optional — enables the depth loss)
└─ masks/              (optional)
```

Binary and text COLMAP models both work, and **Prep** will reshape an export
into this layout for you. Results are written to
`<your folder>/output/` unless you override the output folder, with the final
hierarchy saved as the `.dhier` file named in the parameters.

## Presets

| Preset | Use for |
| --- | --- |
| Default (paper settings) | Faithful reproduction — 60k coarse + 250k fine |
| Fast preview | A rough result in a fraction of the time |
| High quality | Longer schedule, SH degree 3, finer LoD |
| Low VRAM (8–12 GB) | Smaller GPU cache for smaller cards |

Presets set the parameters on tab 3; tune from there.

## Requirements

You need these installed already — the app detects them and tells you which are
missing:

- **NVIDIA GPU** with recent drivers
- **Python 3.10** (64-bit) from python.org, with tcl/tk
- **Git** for Windows
- **CUDA Toolkit** — 12.6 recommended
- **Visual Studio Build Tools** with *Desktop development with C++*

## Notes on this machine

Detected during setup: RTX 3090 (24 GB), CUDA 12.4, MSVC toolset 14.41,
Python 3.10.

Two things this launcher handles automatically that trip up a manual install:

- **The CUDA toolkit must match the PyTorch build.** PyTorch refuses to compile
  extensions when the toolkit's *major* CUDA version differs from the one it was
  built against — a minor difference is only a warning. Setup picks the toolkit
  with an exact wheel match, so a machine with both CUDA 13.2 and 12.8 installed
  gets `cu132` against 13.2 rather than pairing 13.2 with a CUDA 12 wheel. The
  dropdown shows the wheel each toolkit maps to, and setup refuses early with a
  clear message rather than failing deep inside a pip build.
- **Three separate parts of the toolchain police each other's versions**, and
  which one complains depends entirely on the version mix:

  | Guard | Complains when | Cure |
  | --- | --- | --- |
  | nvcc `crt/host_config.h` | MSVC is newer than CUDA knows | `-allow-unsupported-compiler` |
  | MSVC `yvals_core.h` | nvcc is not what the STL expects (*STL1002*) | `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` |
  | CUDA 13's bundled CCCL | MSVC uses its traditional preprocessor | `/Zc:preprocessor` |

  Version tables go stale every release, so setup compiles a representative
  kernel instead, reads the error, applies the remedy that error names, and
  retries — then uses exactly that flag set for the real builds. Locally this
  takes CUDA 11.8 from "will not compile at all" to compiling cleanly against
  MSVC 14.41.

  This check is **advisory**: if it fails for some unrelated reason it prints
  what the compiler actually said, applies the known remedies as a precaution
  and carries on, rather than blocking a build that might have worked. When an
  extension build does fail, the error names the flags that were in effect.
- **The compiler environment is needed at training time, not just at install
  time.** `gsplat` compiles its kernels on first use, so the app runs training
  with the MSVC environment loaded — otherwise the first iteration dies looking
  for `cl.exe`.
- **`gsplat` passes GCC-only flags to MSVC.** Version 1.5.3 hardcodes
  `-Wno-attributes` and `-O3` into its JIT build; `cl.exe` rejects these with
  *"invalid numeric argument"*. Setup rewrites them to MSVC equivalents
  (keeping a `.original` backup) and then compiles the kernels up front, so a
  failure shows up during setup rather than 90 seconds into a training run.
- **Graph view selection cannot read a standard `images.txt`.** Upstream's
  `construct_distance_graph` only treats a line as a camera pose if it splits
  into 11 space-separated fields, but a stock COLMAP pose line has 10. On
  normal COLMAP data nothing matches and it fails with
  *"arrays used as indices must be of integer type"*. Setup patches the parser
  to accept both, which is what makes `graph_view_select` usable at all here.

Both patched files keep a `.original` copy beside them, and the patches are
idempotent — re-running setup will not double-apply them.

- **CMake 4.x rejects the bundled `CMakeLists.txt`**, which asks for
  `cmake_minimum_required(VERSION 3.0)`. The build passes
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to restore the old behaviour.
- **The MSVC environment breaks CMake's Visual Studio generator.** With vcvars
  injected, CMake reports "no CMAKE_CXX_COMPILER could be found" — it wants to
  locate the toolchain itself. So the CUDA extension builds get the vcvars
  environment and the CMake step deliberately does not.

- **The bundled CMakeLists pins `CUDA_ARCHITECTURES "70;75;86"`.** CUDA 13
  dropped sm_70 outright (*"Unsupported gpu architecture 'compute_70'"*), and
  that list covers nothing newer than Ampere — so on a current card the
  hierarchy tool would not run even if it built. Setup retargets it at your
  actual GPU, which also makes the build faster.
- **`ninja` must be findable, not merely installed.** Running the venv's python
  by absolute path does not put its `Scripts` directory on `PATH`, and torch's
  JIT builder shells out to a bare `ninja` — so it fails with *"Ninja is
  required to load C++ extensions"* even with ninja sitting in the venv. Setup
  installs ninja and puts the venv's `Scripts` first on `PATH`.

It also sets `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability, which cuts
build time substantially, and clones with `core.longpaths` enabled because the
bundled Eigen sources exceed Windows' 260-character path limit.

Installed and verified on this machine: torch 2.6.0+cu124, `gsplat`,
`simple_knn`, `gaussian_hierarchy`, `fused_ssim`, and
`GaussianHierarchyCreator.exe`.

## Verified end-to-end

A full run against `test/colmap` (480 images, 6052×4027, 678,750 points)
completed successfully:

| Stage | Result |
| --- | --- |
| Dataset prep | 480 `FULL_OPENCV` cameras → `PINHOLE`, model moved to `sparse/0/` |
| Scene load | 475 train / 5 test, 778,750 points at init |
| Coarse scaffold | 1,000 iterations, 81 MB point cloud |
| View graph | 475 nodes, 47,500 edges |
| Hierarchy | 352 MB `.dhier`, 341 SPTs covering 86% of gaussians |
| Fine training | 3,000 iterations @ 1.39 it/s, loss 0.061, 1.46M gaussians |
| Output | `smoke_test.dhier`, 169 MB |

Peak GPU memory was 1.4 GB with `cache_size` at its default, so there is plenty
of headroom on a 24 GB card for the full-length schedule.

## Things that will bite you

- **No spaces in paths.** Upstream `train.py` builds an unquoted shell command
  for the hierarchy step, so a space anywhere in the dataset or install path
  breaks it. The app flags this before you start.
- **Graph view selection needs >100 images.** It fits a k-nearest-neighbour
  graph over cameras with k hardcoded to 100 upstream, so smaller captures
  cannot supply enough points. The app counts your images (allowing for the
  holdout stride) and offers to switch it off when you press Start.
- **`cache_size` is the VRAM dial.** If you hit out-of-memory, lower it first.
- **Resume works, but watch the scaffold length.** Leave *skip the coarse pass
  if a scaffold already exists* ticked and an interrupted run picks up from the
  existing scaffold. If that scaffold is shorter than the coarse pass you just
  configured, the app asks whether to reuse it or retrain — otherwise a quick
  test run silently caps the quality of every later run against the same
  output folder.

## Layout

```
lod_trainer.py           the app
Launch LoD Trainer.bat   double-click entry point
settings.json            your saved settings (created on first use)
repo/                    cloned upstream source (created by setup)
venv/                    isolated Python environment (created by setup)
torch_extensions/        JIT-compiled CUDA kernels (created by setup)
```

To start completely fresh, delete `repo/`, `venv/` and `torch_extensions/`, then
run setup again.

## Relationship to upstream

This repository contains only the launcher. It does not vendor the research
code — setup clones
[FelixWindisch/LoDOfGaussians](https://github.com/FelixWindisch/LoDOfGaussians)
into `repo/` at install time and patches it there, so upstream stays the single
source of truth and you always build against current code.

For reference, the upstream project's own documentation and license are kept
here as [README.upstream.md](README.upstream.md) and
[LICENSE.upstream.md](LICENSE.upstream.md).

The method, the training code, and all the research are Felix Windisch's and
their co-authors' work (SIGGRAPH 2026). Upstream is free for non-commercial,
research and evaluation use — those terms govern anything you train with this,
so read `LICENSE.upstream.md` before using results commercially.
