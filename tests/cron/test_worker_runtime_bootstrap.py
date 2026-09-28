"""External workers select dependencies before importing cron and retain their lease.

The store interpreter fixture has no dependencies. Only the committed generation
provides the sentinel; a test runner with ruamel installed cannot mask a missing
bootstrap. PM activation, scheduler entry, ownership transfer and GC are real.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
from types import SimpleNamespace
import venv

import pytest


REPO = Path(__file__).resolve().parents[2]
SENTINEL = "cron_generation_sentinel"


def _wait_for_file(path, process):
    deadline = time.monotonic() + 20
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"worker exited {process.poll()} before publishing {path.name}"


@pytest.fixture
def worker_layout(tmp_path, monkeypatch):
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from tools import process_registry

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("HERMES_MODEL", "fixture-launch")
    (home / ".env").write_text("HERMES_MODEL=fixture-launch\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-launch-only")
    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK"):
        monkeypatch.setenv(name, "1")
    previous = is_multiplex_active()
    set_multiplex_active(True)
    # Exercise the production spawn and handoff without contacting a service manager.
    monkeypatch.setattr(
        process_registry, "restart_safe_gateway_child_argv",
        lambda argv, **kw: process_registry.GatewayChildDispatch(argv=argv, mode="degraded"),
    )
    profiles = []
    for name in ("a", "b"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text(json.dumps({
            "model": {"default": f"fixture-{name}"},
            "cron": {"script_timeout_seconds": 30},
        }), encoding="utf-8")
        (profile / ".env").write_text(f"HERMES_MODEL=fixture-{name}\n", encoding="utf-8")
        profiles.append(profile)
    try:
        yield SimpleNamespace(home=home, profiles=profiles, store=tmp_path / "store")
    finally:
        set_multiplex_active(previous)


def _generation(name):
    from pm.environments import install_state_dir, site_packages

    environment = install_state_dir(REPO) / "environments" / name / "venv"
    site = site_packages(environment)
    site.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}\n", encoding="utf-8")
    (environment.parent / ".lease-managed").touch()
    # Borrow only the suite's dependencies, not its interpreter or bootstrap.
    # The unique sentinel exists exclusively inside this selected generation.
    (site / "worker-fixture.pth").write_text(
        f"{sysconfig.get_paths()['purelib']}\nimport {SENTINEL}\n", encoding="utf-8")
    (site / f"{SENTINEL}.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\nimport ruamel.yaml\n"
        "home = Path(os.environ['HERMES_HOME'])\n"
        "report = {'sentinel': __file__, 'python': sys.executable, 'pid': os.getpid(),\n"
        "          'yaml': ruamel.yaml.__file__, 'home': str(home)}\n"
        "(home / ('bootstrap-' + str(os.getpid()) + '.json')).write_text(json.dumps(report))\n",
        encoding="utf-8",
    )
    return environment


def _select(environment):
    from hermes_cli.runtime_state import runtime_lock
    from pm.environments import runtime_facts_path
    from pm.filesystem import durable_write_bytes

    with runtime_lock(REPO) as held:
        assert held
        durable_write_bytes(runtime_facts_path(REPO), json.dumps({
            "packages": {"venv": {"environment": str(environment)}}
        }).encode())


def _launch_probe(profile, monkeypatch, inspect_running):
    import cron.executions as executions
    import cron.scheduler as scheduler
    from cron.jobs import create_job
    from cron.scheduler_provider import _profile_cron_scope

    started = profile / "started.json"
    release = profile / "release"
    started.unlink(missing_ok=True)
    release.unlink(missing_ok=True)
    scripts = profile / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "probe.py").write_text(
        "import json, os, time\nfrom pathlib import Path\n"
        "config = json.loads((Path(os.environ['HERMES_HOME']) / 'config.yaml').read_text())\n"
        f"Path({str(started)!r}).write_text(json.dumps({{\n"
        "    'home': os.environ.get('HERMES_HOME'), 'model': os.environ.get('HERMES_MODEL'),\n"
        "    'configured_model': config['model']['default'],\n"
        "    'foreign_key': os.environ.get('OPENAI_API_KEY'), 'parent': os.getppid()}))\n"
        "deadline = time.monotonic() + 25\n"
        f"while not Path({str(release)!r}).exists():\n"
        "    assert time.monotonic() < deadline, 'fixture release timed out'\n"
        "    time.sleep(0.02)\n",
        encoding="utf-8",
    )
    real_popen = subprocess.Popen
    real_wait = scheduler._wait_for_external_cron_worker
    children = []
    observed = []

    def spawn(argv, **kwargs):
        env = kwargs["env"]
        assert kwargs["cwd"] == str(REPO)
        assert env["HERMES_HOME"] == str(profile)
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO)
        assert sysconfig.get_paths()["purelib"] not in env["PYTHONPATH"].split(os.pathsep)
        assert not {"VIRTUAL_ENV", "OPENAI_API_KEY", "HERMES_INTERACTIVE",
                    "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK"}.intersection(env)
        child = real_popen(argv, **kwargs)
        children.append(child)
        return child

    def wait(child, **kwargs):
        row = executions.get_execution(kwargs["execution_id"])
        assert row["status"] == "running" and row["pid"] == child.pid
        # This mapping is filled from the real acknowledgement, before the waiter.
        assert scheduler._running_worker_pids[scheduler._inflight_key(job["id"])] == child.pid
        assert not list((profile / "cron" / "external-workers").glob("*.json"))
        _wait_for_file(started, child)
        report = json.loads(started.read_text())
        assert report == {"home": str(profile), "model": None,
                          "configured_model": f"fixture-{profile.name}",
                          "foreign_key": None, "parent": child.pid}
        inspect_running(child)
        observed.append(child.pid)
        release.touch()
        result = real_wait(child, **kwargs)
        assert child.wait(timeout=20) == 0
        return result

    with _profile_cron_scope(profile), monkeypatch.context() as patch:
        job = create_job(prompt=None, schedule="every 1h", script="probe.py",
                         no_agent=True, deliver="local")
        job["execution_id"] = executions.create_execution(job["id"], source="direct")["id"]
        patch.setattr(scheduler.subprocess, "Popen", spawn)
        patch.setattr(scheduler, "_wait_for_external_cron_worker", wait)
        try:
            assert scheduler._launch_external_cron_worker(job)
            assert observed
            assert executions.get_execution(job["execution_id"])["status"] == "completed"
        finally:
            release.touch()
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            scheduler.release_running_job(job["id"], home=profile)


@pytest.mark.platforms("posix")
def test_pm_worker_bootstraps_before_cron_and_leases_generation_through_rotation(
    worker_layout, monkeypatch,
):
    import cron.scheduler as scheduler
    from hermes_cli.runtime_state import collect_generations, leases_held

    layout = worker_layout
    bare = layout.store / "bare"
    venv.EnvBuilder(with_pip=False).create(bare)
    python = bare / "bin" / "python3"
    (layout.store / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "bare"}}}), encoding="utf-8")
    # Model the gateway's bare sys.executable without modifying the launcher's
    # resolver: it must read the real store fact above.
    monkeypatch.setattr(scheduler, "sys", SimpleNamespace(executable=str(python)))
    first, second = _generation("first"), _generation("second")
    _select(first)
    from pm.environments import site_packages
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO), str(site_packages(first))]))
    raw = subprocess.run([str(python), "-I", "-c", f"import {SENTINEL}"],
                         capture_output=True, text=True, timeout=20)
    assert raw.returncode != 0 and f"No module named '{SENTINEL}'" in raw.stderr
    raw = subprocess.run([str(python), "-m", "cron.scheduler",
                          "--external-worker-file", str(layout.home / "missing"),
                          "--ack-file", str(layout.home / "never.ready")],
                         env=dict(os.environ, PYTHONPATH=str(REPO)), cwd=layout.home,
                         capture_output=True, text=True, timeout=20)
    assert raw.returncode != 0 and "No module named 'ruamel'" in raw.stderr
    assert not (layout.home / "never.ready").exists()

    for index, profile in enumerate([*layout.profiles, layout.profiles[0]]):
        expected = first if index == 0 else second

        def inspect(child):
            report = json.loads((profile / f"bootstrap-{child.pid}.json").read_text())
            assert Path(report["sentinel"]).is_relative_to(expected)
            assert report["python"] == str(python)
            assert report["home"] == str(profile) and report["pid"] == child.pid
            assert leases_held(expected.parent)
            if index == 0:
                _select(second)
                assert first.parent not in collect_generations(REPO, min_age_seconds=0)
                assert first.exists() and second.exists()

        _launch_probe(profile, monkeypatch, inspect)
        assert not leases_held(expected.parent)
        if index == 0:
            assert first.parent in collect_generations(REPO, min_age_seconds=0)
            assert not first.exists() and second.exists()


@pytest.mark.platforms("posix")  # The probe asserts native parent/worker PID identity.
def test_non_pm_worker_keeps_external_interpreter_and_profile_isolation(worker_layout, monkeypatch):
    from hermes_cli._launchers import resolve_store_python
    from pm.environments import committed_venv, install_state_dir

    assert resolve_store_python(REPO) is None and committed_venv(REPO) is None
    def inspect(child):
        assert child.args[0] == sys.executable

    for profile in [*worker_layout.profiles, worker_layout.profiles[0]]:
        _launch_probe(profile, monkeypatch, inspect)
    assert not install_state_dir(REPO).exists()
