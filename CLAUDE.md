# CLAUDE.md

## jaxguam GPU/CUDA fix (2026-06-10)

- **Fixed**: jaxguam container was running JAX on CPU
  (`jax.devices()` -> `[CpuDevice(id=0)]`) because (a) the Dockerfile
  installed plain `jax jaxlib` (CPU-only wheels), and (b) the `jaxguam`
  service in `docker/docker-compose.yml` had no GPU device reservation, so
  `libcuda.so` was never mounted into the container by the NVIDIA Container
  Toolkit.
  - `vehicles/jaxguam/Dockerfile`: `jax jaxlib` -> `"jax[cuda12]"`.
  - `docker/docker-compose.yml`: added `deploy.resources.reservations.devices`
    nvidia block + `NVIDIA_VISIBLE_DEVICES`/`NVIDIA_DRIVER_CAPABILITIES` env
    to the `jaxguam` service (matches `env_sim`).
  - Container was force-recreated (`docker compose up -d --force-recreate
    jaxguam`) and `jax[cuda12]==0.6.2` was live pip-installed to verify ->
    `jax.devices()` now returns `[CudaDevice(id=0)]`.
  - **Caveat**: the live pip install is NOT baked into the image. If
    `jaxguam` is recreated again before rebuilding the image
    (`docker compose build jaxguam`), it'll revert to CPU-only jaxlib and
    need `pip install -U "jax[cuda12]==0.6.2"` re-run inside the container
    (conda env `py310`).

## Current state / pending follow-ups (2026-06-10)

- **GPU reboot fixed the wedged CUDA driver.** `torch.cuda.is_available()`
  now returns `True`. The earlier `RuntimeError: CUDA unknown error` /
  "CUDA-capable device(s) is/are busy or unavailable" is resolved.

- **Vanilla YOLO config smoke test: PASSED.**
  `configs/hyp_bayes_default.yaml` is the "vanilla" baseline — same
  project/run setup (data, model, epochs=50, batch=50, workers, etc.) as
  `configs/hyp_bayes.yaml`, but with all augmentation and optimizer
  hyperparameters reset to Ultralytics defaults (`optimizer: AdamW`,
  `lr0/lrf/momentum/weight_decay` = stock, `hsv_*`/`mosaic`/`mixup`/etc. =
  stock, `deterministic: true`, `seed: 0`, `amp: true`).
  - Ran via `/tmp/run_vanilla_test.py` with `scale=0.5`, `hsv_v=0.4`,
    output in `./vanilla_test_run/`.
  - Result: early-stopped at 35/50 epochs (best at epoch 25, patience=10),
    ~9.3 min total. Best model: `vanilla_test_run/train/weights/best.pt`.
  - Final val metrics: P=0.798, R=0.731, mAP50=0.748, mAP50-95=0.365.

## Session changes (2026-06-10)

- **Landing controller bbox selection reverted to last-row tracking.**
  `vehicles/jaxguam/script/landing_controller_v2.py` and
  `landing_controller.py` (`the_obj = ...`) reverted from
  `np_tracking[np.argmin(np_tracking[:, 4])]` (earliest/lowest SORT
  track-id) back to `np_tracking[-1]`. Rationale: spurious detections
  spawn new low-track-id tracks that pull the UAV off-target, hurting
  success rate. (Requires a `jaxguam` service restart to take effect —
  these are persistent `rosrun` processes.)

- **Altitude-dependent xy spawn offset.**
  `vehicles/jaxguam/script/run_simulator.py`: per-trial init x/y offset is
  now `max_xy_offset = ALT_XY_RATIO * (init_z - ideal_z)`, `ALT_XY_RATIO =
  0.6` (inspired by `RANGE_BY_ALT` in `~/works/rraaa-sim/POMCP_CARLA`).
  Keeps the helipad within the 90°-FOV down-camera (640x640) at all spawn
  altitudes, with margin. Result still clamped to
  `range_offset.x/y` from `configs/single-static.yml`.

- **RNG seeding decoupled from (scale, hsv_v).**
  `run_simulator.py` now seeds `np.random` via
  `sim_seed = (base_seed + run_num) % 2**32`, where `base_seed =
  bayesian_optimisation.seed` (42) and `run_num` is parsed from
  `metadata['run_number']` (e.g. "run_341" -> 341). Scenario set (z-strata +
  init x/y) is reproducible per-BO-iteration across reruns of the same BO
  trajectory, but varies from one BO query to the next — independent of the
  proposed `(scale, hsv_v)`.

- **`m_trials` unified with `stratification`.**
  `configs/single-static.yml`'s standalone `bayesian_optimisation.m_trials`
  field removed. `bo_optimizer.py` now derives `M_TRIALS = n_strata *
  samples_per_stratum` from `stratification` — single source of truth,
  matches `run_simulator.py`'s actual trial count. Currently `n_strata: 2,
  samples_per_stratum: 5` -> `M_TRIALS = 10`.

- **`episode_timeout` bumped 150 -> 170s** in `configs/single-static.yml`.

- **YOLO training reverted to cold-starting from `yolov8s.pt` every BO
  iteration.** The warm-start experiment (train a (scale=0.5, hsv_v=0.5)
  reference checkpoint at the start of every launch, then fine-tune every BO
  iteration from that fixed checkpoint) was reverted at the user's request
  ("Now, I want to cold start with yolov8s like before").
  - `bo_optimizer.py` (`__main__`, before the SNAPS section): the
    "Train (scale=0.5, hsv_v=0.5) reference model for warm-starting" block
    removed; `BASE_WEIGHTS = 'yolov8s.pt'` is now set directly (still
    forwarded through `partial(oracle_yolo, ..., base_weights=BASE_WEIGHTS)`
    to `YOLOTrainingPipeline`, which still supports `base_weights` for any
    future use).
  - `yolo_training/YOLO_training_pipeline.py`'s `base_weights='yolov8s.pt'`
    param and `oracle_yolo()`'s `base_weights` param are unchanged (left in
    place — they're a no-op pass-through of `'yolov8s.pt'` now, same as the
    pre-warm-start behavior).

- **"Stuck / no-progress" trials now fail early instead of running the full
  170s `episode_timeout`.** At high altitude, the UAV can get stuck
  oscillating between multiple false detections — nonzero velocity
  corrections that swing it back and forth without net progress, so the
  existing `zero_vel_stop_sec` (5s) check never fires (see run_343 rows
  3/7/10 for the *zero*-velocity case, which already stops early).
  - `configs/single-static.yml`: added `stuck_window_sec: 15`,
    `stuck_displacement_m: 3`, `stuck_min_altitude_m: 100`.
  - `vehicles/jaxguam/script/run_simulator.py`: `SimulationMonitor` now
    keeps a rolling `position_history` deque (last `stuck_window_sec`
    seconds, populated in `pose_callback`) and a new `is_stalled()` method —
    returns `True` only when `current_pose.z > stuck_min_altitude_m` AND a
    full window of history exists AND net 3D displacement vs. the oldest
    point in the window is `< stuck_displacement_m`. `run_simulation()`'s
    main loop checks `monitor.is_stalled()` between the `zero_vel` and
    `timeout` checks, with `stop_reason = "No progress (<3m) over 15s"` —
    flows into `simulation_results.csv`/success classification exactly like
    the other early-stop reasons (lands far from target -> `Fail`).
  - **Requires a `jaxguam` service restart** (persistent `rosrun` process)
    to take effect.
  - **Not yet verified**: run a trial and confirm `simulation_results.csv`
    shows the new stop reason with `Time to Land` ~15-20s for previously
    170s-timeout high-altitude stuck trials, and that normal trials (e.g.
    run_343 row 2, 168s success) are unaffected.

- **`stuck_window_sec` increased 15 -> 30s** in `configs/single-static.yml`
  (per user request to give the no-progress check a longer window before
  declaring a high-altitude trial stalled).

- **Z stopping/landing criteria set to a single `z <= ideal_z` threshold (not
  a +/- band).** `vehicles/jaxguam/script/run_simulator.py`'s
  `pose_callback()`:
  - `z_value_below_threshold` (drives the "landed" stop check):
    `msg.pose.position.z <= ideal_z`; `stop_reason = f"Z dropped below
    {ideal_z}m"`.
  - `within_tolerance`'s z condition: `p.z <= ideal_z` (combined with the
    existing x/y `TOL_X`/`TOL_Y` checks).
  - The final `success` formula (`abs(landing_pose.z - ideal_z) < TOL_Z`)
    is unchanged — only the *stop* condition uses a single descent
    threshold rather than a +/- TOL_Z band. (This was toggled back and
    forth a couple times this session — this is the final state.)
    **Requires a `jaxguam` service restart** to take effect.

- **"Init failed" trials (sim never spawned the UAV at its init pose) are now
  retried instead of counted as a Fail.** Seen in run_344 trial 1: vehicle
  was stuck at the default CARLA spawn `(-125, 210, 150)` (= `ego_vehicle.
  location` in `configs/single-static.yml`) and never reached its requested
  per-trial init pose within `init_gating.wait_timeout` (20s) — a CARLA
  cold-start hiccup, not a model-performance result, but it was being logged
  as a `Fail` and consuming one of the `m_trials` slots.
  - `configs/single-static.yml`: added `init_gating.max_retries: 2`.
  - `vehicles/jaxguam/script/run_simulator.py`: added
    `MAX_INIT_RETRIES = config['init_gating'].get('max_retries', 2)`. The
    per-trial loop now retries `run_simulation(...)` (fresh `land_vtol.py`
    subprocess each time) up to `MAX_INIT_RETRIES + 1` total attempts while
    `stop_reason.startswith("Init failed")`, popping the bookkeeping
    (`final_positions`/`final_euler_angles`) appended by each failed
    attempt so `simulation_results.csv` stays one-row-per-trial. If all
    retries fail, the trial is recorded as `Init failed`/`Fail` as before
    (no infinite loop).
  - **Requires a `jaxguam` service restart** to take effect.
  - **Not yet verified**.

## Session changes (2026-06-11)

- **`configs/hyp_bayes.yaml` reverted to match Train231's hyperparameters**
  (the resolved config in `training_result/Train231/temp_config.yaml`).
  Changed: `optimizer` AdamW->Adam, `half` false->true, `augment`
  false->true, `lr0` 0.01->0.0005, `lrf` 0.01->0.05, `weight_decay`
  0.0005->0.001, `hsv_s` 0.7->0.9, `hsv_v` 0.4->0.5, `scale` 0.5->0.1,
  `mixup` 0.0->0.7. `project`/`save_dir` left untouched (pipeline-managed
  paths). `scale`/`hsv_v` are placeholders overwritten per-query by the BO
  loop anyway.
  - **Caveat**: this was edited while a BO run (run28/Train290, optimization
    run 28) was active — `bo_optimizer.py` reads `hyp_bayes.yaml` fresh each
    iteration, so any iteration starting after this edit picked up the new
    hyperparameters mid-run (inconsistent with earlier iterations of the
    same run).

- **Added `plots/plot_trained_points_1d.py`** — 1D strip plot of trained
  YOLO models' `(Scale, HSV_V)` values from `utils/performance_summary.csv`
  (one point per `Training Folder`, success rate averaged across repeated
  test runs), colored by Success Rate. Output: `plots/trained_points_1d.png`.

## Session changes (2026-07-03)

- **Added `spectator_control.py` (repo root)** — keyboard free-fly control of
  the CARLA spectator camera, for driving the view over remote desktop
  without a mouse. A small pygame window (reads keys via SDL when focused, so
  it works over RDP/VNC, no sudo) drives `world.get_spectator()`:
  Pitch `I/K`, Yaw `J/L`, Roll `U/O`, Move `WASD`, Up/Down `Q/E`, `Shift`=4x
  boost, `P`=snapshot, `ESC`=quit. Connects `localhost:2000` by default
  (`python spectator_control.py [host] [port]`).
  - **Must live at repo root, NOT in `utils/`.** Running it from inside
    `utils/` puts that dir on `sys.path[0]`, and `utils/logging.py` then
    shadows the stdlib `logging` module -> circular-import crash on
    `import pygame`. (Was originally created in `utils/`, hit this, moved.)
  - Per-frame RPC calls are wrapped in `try/except RuntimeError` + a 20s
    client timeout so a briefly-busy sim skips a frame instead of aborting.
    NOTE: a CARLA C++ `TimeoutException` does NOT always surface as a Python
    `RuntimeError`, so the process can still SIGABRT (exit 134) on a hard
    timeout — not fully hardened.
  - Must run on the **same X display as CARLA (`:1`)**, i.e. inside the
    remote-desktop session. `pip install pygame` if missing; `carla` client
    already present on the host.

- **`P`-key snapshot captures the real CarlaUE4 window, not a spawned
  camera.** First tried a temporary `sensor.camera.rgb` at the spectator
  transform, but (a) it didn't match the window's FOV/post-processing and
  (b) `spectator.get_transform()` read `(0,0,0)` at rest so it framed the
  wrong spot. Final approach: `take_snapshot()` finds the CARLA X11 window by
  title substring `"CarlaUE4"` via `xwininfo -root -tree`, then grabs its
  exact pixels with ImageMagick `import -window <id>`. Saves to
  `./snapshots/carla_<timestamp>.png` (dir auto-created, relative to CWD).
  Works despite CARLA rendering via Vulkan.
  - **Snapshot resolution == the CARLA window's current on-screen size**
    (e.g. was 1387x557 when small, 1850x1016 when enlarged). It is NOT the
    launch `-ResX=1920 -ResY=1080`; window borders/title-bar shrink it below
    1920x1080. For a true 1920x1080 grab, fullscreen the window (Alt+Enter /
    F11) or relaunch CARLA `-fullscreen`. For higher-than-window res, use the
    UE4 console (`~`) command `HighResShot 3840x2160` -> saved under
    `CarlaUE4/Saved/Screenshots/`.
  - Requires `xwininfo` + `import` (ImageMagick) — both present on host.
    `xdotool`/`wmctrl`/`python-xlib` are NOT installed, so the window can't
    be resized programmatically without a `sudo apt install`.

- **CARLA runtime facts (this host).** Native `CarlaUE4.sh` runs on the host
  (not in a container) at `~/works/carla`, launched `-windowed -ResX=1920
  -ResY=1080 -carla-server -fps=20`, Vulkan (`SF_VULKAN_SM5`), display `:1`,
  simulator API `0.9.15-330`. Containers (`env_sim_ue4`, `jaxguam`,
  `roscore`, `yolov5`) are **host-networked**, so their `127.0.0.1:2000`
  hits the same native CARLA. `env_sim_ue4` runs `bash` as entrypoint; ROS
  nodes are launched manually inside via `roslaunch rraaa run.launch`
  (node_carla.py, node_input_display.py, octomap_server).

- **Ego-vehicle spawn ownership / config flow (clarified, not changed).**
  The CARLA spawn is done by `env_sim`'s `node_carla.py:82` ->
  `Environment(...).start()` -> `spawn_ego_vehicle()`
  (`env_sim/rraaa/script/tools/environment.py:1198`), which reads
  `ego_vehicle.location` and spawns with `world.spawn_actor`. `node_vehicle`
  (jaxguam) does NOT spawn into CARLA — it's the GUAM dynamics node.
  - Both nodes read the **flattened `constants.merged_config_path`** that
    `rraaa.py` writes at launch, NOT `single-static.yml` directly. Editing
    `single-static.yml` only takes effect after `rraaa.py` regenerates the
    merged config (relaunch the stack); restarting a single node re-reads the
    stale merged file.
  - In the BO/trial path, `run_simulator.py:215` OVERWRITES
    `ego_vehicle.location.x/y/z` per trial with the sampled init pose (from
    `ideal_position` + `range_offset` + `stratification`), so hand-editing
    `ego_vehicle.location` has no effect there. The user commented out that
    overwrite so the spawn follows `single-static.yml` (verified: ego spawns
    at the exact configured location).

- **Ego spawn rotation is hardcoded at `environment.py:1225`**, and the
  config's `ego_vehicle.rotation` block is ignored. `carla.Rotation` arg
  order is `(pitch, yaw, roll)`. This session it was toggled to `(0,0,180)`
  (roll=180) then `(0,180,0)` (yaw=180) on request, and finally **reverted to
  `carla.Rotation(0, 0, 0)`** — current state. A proposed edit to read the
  rotation from config (like `location`) was rejected; it stays hardcoded.
  Requires an `env_sim_ue4` / `node_carla` restart to take effect.
