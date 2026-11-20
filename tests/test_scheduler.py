"""Tests for the single compute slot and the urgent hand-over policy.

The scheduler is exercised directly (deterministic thread choreography) and
end to end through the HTTP API with concurrent real inference calls.
"""

from __future__ import annotations

import io
import json
import threading
import time
from collections import deque

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_mod
from app.inference import parse_model_params, reference_full_inference
from app.main import app
from app.scheduler import ComputeScheduler


def _params(seed: int = 5) -> bytes:
    rng = np.random.default_rng(seed)
    return json.dumps(
        {
            "layers": [
                {"kernel": rng.normal(size=(3, 3)).tolist(), "bias": 0.25},
                {"kernel": rng.normal(size=(3, 3)).tolist(), "bias": -0.25},
            ]
        }
    ).encode()


def _npy(image: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    np.save(buf, image)
    buf.seek(0)
    return buf


def _post(root="/segment", *, image, urgent=None, params=None, client=None):
    files = {
        "image": ("img.npy", _npy(image), "application/octet-stream"),
        "params": ("p.json", params or _params(), "application/json"),
    }
    data = {}
    if urgent is not None:
        if isinstance(urgent, bool):
            data["urgent"] = "true" if urgent else "false"
        else:
            data["urgent"] = urgent
    return (client or TestClient(app)).post(root, files=files, data=data)


# ---------------------------------------------------------------------------
# HTTP-level validation of the urgent flag
# ---------------------------------------------------------------------------


def test_urgent_form_field_in_docs():
    schema = TestClient(app).get("/openapi.json").json()
    body = schema["paths"]["/segment"]["post"]["requestBody"]
    ref = body["content"]["multipart/form-data"]["schema"]["$ref"]
    form = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    assert "urgent" in form["properties"]
    assert form["properties"]["urgent"]["default"] is False
    # The original upload fields are unchanged.
    assert {"image", "params"} <= set(form["properties"])
    assert set(form["required"]) == {"image", "params"}


def test_urgent_within_limit_is_accepted_and_correct():
    rng = np.random.default_rng(1)
    for shape in ((1, 1), (1024, 1024), (1024, 3), (17, 1024)):
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        resp = _post(image=image, urgent=True)
        assert resp.status_code == 200, (shape, resp.text)
        mask = np.load(io.BytesIO(resp.content))
        layers = parse_model_params(_params())
        assert np.array_equal(mask, reference_full_inference(image, layers))


@pytest.mark.parametrize("shape", [(1025, 1), (1, 1025), (1025, 1025), (2048, 8)])
def test_urgent_over_limit_rejected_without_computing(shape):
    image = np.zeros(shape, dtype=np.uint8)
    resp = _post(image=image, urgent=True)
    assert resp.status_code == 400
    assert "urgent" in resp.json()["detail"]
    assert "1024" in resp.json()["detail"]


def test_non_urgent_large_image_still_allowed():
    # The limit applies only to urgent jobs; the normal path is unchanged.
    resp = _post(image=np.zeros((1025, 2), dtype=np.uint8), urgent=False)
    assert resp.status_code == 200


def test_original_request_without_field_still_works():
    resp = _post(image=np.zeros((9, 7), dtype=np.uint8), urgent=None)
    assert resp.status_code == 200
    mask = np.load(io.BytesIO(resp.content))
    assert mask.shape == (5, 4)


def test_invalid_urgent_flag_rejected_by_form_parsing():
    resp = _post(image=np.zeros((4, 4), dtype=np.uint8), urgent="maybe")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Scheduler unit tests (deterministic)
# ---------------------------------------------------------------------------


def _run(target, *args):
    errors: list[BaseException] = []

    def wrapper():
        try:
            target(*args)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    return t, errors


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def test_single_owner_excludes_everybody_else():
    sched = ComputeScheduler()
    release = threading.Event()
    inside: list[str] = []
    t1, e1 = _run(lambda: _hold(sched, "a", release, inside))
    assert _wait_for(lambda: inside == ["a"])
    t2, e2 = _run(lambda: _hold(sched, "b", release, inside))
    time.sleep(0.05)
    assert inside == ["a"]
    release.set()
    t1.join(5)
    t2.join(5)
    assert not e1 and not e2
    # Exactly one at a time: b entered only after a released.
    assert inside == ["a", "b"]


def _hold(sched, name, release, inside):
    t = sched.ticket()
    sched.acquire(t)
    try:
        inside.append(name)
        release.wait(5)
    finally:
        sched.release(t)


def test_normal_checkpoint_does_not_yield_without_urgent():
    sched = ComputeScheduler()
    t = sched.ticket()
    sched.acquire(t)
    try:
        sched.checkpoint(t)  # returns immediately, stays owner
        assert sched._owner is t
        assert t.yielded is False
    finally:
        sched.release(t)
    # Even a later urgent job is then served normally.
    u = sched.ticket(urgent=True)
    sched.acquire(u)
    sched.release(u)


def test_yield_happens_at_most_once_per_normal_job():
    sched = ComputeScheduler()
    normal = sched.ticket()
    sched.acquire(normal)
    order: list[str] = []
    urgent_running = threading.Event()
    finish_urgent = threading.Event()
    try:
        urgent = sched.ticket(urgent=True)

        def urgent_user():
            sched.acquire(urgent)
            order.append("urgent-running")
            urgent_running.set()
            finish_urgent.wait(5)
            sched.release(urgent)
            order.append("urgent-released")

        urgent_thread, urgent_errors = _run(urgent_user)
        assert _wait_for(lambda: sched._urgent == [urgent])

        def hand_over():
            sched.checkpoint(normal)
            order.append("normal-resumed")

        waiter, errors = _run(hand_over)
        assert urgent_running.wait(5)
        assert sched._owner is urgent
        finish_urgent.set()
        urgent_thread.join(5)
        waiter.join(5)
        assert not errors and not urgent_errors
        assert order == ["urgent-running", "urgent-released", "normal-resumed"]
        assert normal.yielded is True
        # A second urgent job cannot make it yield again.
        late = sched.ticket(urgent=True)
        assert late.seq > urgent.seq
        late_thread, _ = _run(lambda: (sched.acquire(late), sched.release(late)))
        assert _wait_for(lambda: sched._urgent == [late])
        sched.checkpoint(normal)
        assert sched._owner is normal
        assert sched._urgent == [late]
        # Clean up the queued late job so the monitor is left idle.
        sched.release(normal)
        late_thread.join(5)
    finally:
        sched.release(normal)


def test_urgent_jobs_are_served_in_acceptance_order():
    sched = ComputeScheduler()
    normal = sched.ticket()
    sched.acquire(normal)
    try:
        served: list[int] = []

        def urgent_user(ticket, tag):
            sched.acquire(ticket)
            served.append(tag)
            sched.release(ticket)

        threads = []
        # Enqueue in order; waiting for each ticket to reach the queue makes
        # the acceptance order deterministic even though the threads race.
        for i in range(4):
            ticket = sched.ticket(urgent=True)
            th, errs = _run(urgent_user, ticket, i)
            threads.append(th)
            assert _wait_for(lambda: len(sched._urgent) + len(served) > i)
        waiter, errs = _run(lambda: sched.checkpoint(normal))
        for th in threads:
            th.join(5)
        waiter.join(5)
        assert not errs
        assert served == [0, 1, 2, 3]
        assert sched._owner is normal
    finally:
        sched.release(normal)


def test_normal_jobs_queue_fifo_and_urgent_jumps_ahead():
    sched = ComputeScheduler()
    first = sched.ticket()
    sched.acquire(first)
    try:
        second = sched.ticket()
        order = []

        def user(tag_ticket, tag):
            sched.acquire(tag_ticket)
            order.append(tag)
            sched.release(tag_ticket)

        t2, _ = _run(user, second, "normal-2")
        assert _wait_for(lambda: len(sched._normal) == 1)
        third = sched.ticket()
        t3, _ = _run(user, third, "normal-3")
        assert _wait_for(lambda: len(sched._normal) == 2)
        # Urgent arrives after both normals are already waiting.
        urgent = sched.ticket(urgent=True)
        tu, _ = _run(user, urgent, "urgent")
        assert _wait_for(lambda: len(sched._urgent) == 1)
        sched.release(first)
        t2.join(5)
        t3.join(5)
        tu.join(5)
        assert order == ["urgent", "normal-2", "normal-3"]
    finally:
        sched.release(first)


def test_urgent_arriving_after_yield_does_not_block_resume():
    sched = ComputeScheduler()
    normal = sched.ticket()
    sched.acquire(normal)
    order: list[str] = []
    early_running = threading.Event()
    finish_early = threading.Event()
    try:
        early = sched.ticket(urgent=True)

        def early_user():
            sched.acquire(early)
            order.append("early")
            early_running.set()
            finish_early.wait(5)
            sched.release(early)

        early_thread, _ = _run(early_user)
        assert _wait_for(lambda: sched._urgent == [early])

        def hand_over():
            sched.checkpoint(normal)
            order.append("normal-resumed")

        waiter, errs = _run(hand_over)
        assert early_running.wait(5)

        # A second urgent job is accepted while the normal job has yielded
        # but has not resumed yet.
        late = sched.ticket(urgent=True)

        def late_user():
            sched.acquire(late)
            order.append("late")
            sched.release(late)

        late_thread, _ = _run(late_user)
        assert _wait_for(lambda: len(sched._urgent) == 1)
        finish_early.set()
        early_thread.join(5)
        waiter.join(5)
        assert not errs
        # The normal job resumes before the late arrival: it only waits for
        # the urgent work accepted before it yielded.
        assert order == ["early", "normal-resumed"]
        assert sched._owner is normal
        sched.release(normal)
        late_thread.join(5)
        assert order == ["early", "normal-resumed", "late"]
    finally:
        sched.release(normal)


# ---------------------------------------------------------------------------
# End-to-end concurrency: urgent preempts a running normal job
# ---------------------------------------------------------------------------


@pytest.fixture()
def instrumented(monkeypatch):
    """Force small tiles and record *actual* concurrent tile computation.

    A job parked at a hand-over checkpoint is not computing; the counters are
    dropped around the checkpoint call and restored only once it returns.
    """
    real = main_mod.tiled_segment
    state = {"computing": 0, "max_computing": 0, "tiles": 0}
    lock = threading.Lock()

    def wrapped(image_path, layers, feat_path, out_path, tile=512,
                checkpoint=None):
        def enter_compute():
            with lock:
                state["computing"] += 1
                state["max_computing"] = max(
                    state["max_computing"], state["computing"]
                )

        def leave_compute():
            with lock:
                state["computing"] -= 1

        enter_compute()  # a job that owns the slot is computing
        try:
            def counted():
                if checkpoint is None:
                    return
                with lock:
                    state["tiles"] += 1
                time.sleep(0.005)  # keep jobs long enough to interleave
                leave_compute()
                checkpoint()  # may park here: not computing meanwhile
                enter_compute()

            return real(
                image_path, layers, feat_path, out_path, tile=16,
                checkpoint=counted,
            )
        finally:
            leave_compute()

    monkeypatch.setattr(main_mod, "tiled_segment", wrapped)

    def snapshot():
        with lock:
            return state["tiles"]

    state["snapshot"] = snapshot
    return state


def test_urgent_finishes_before_normal_and_both_match_reference(instrumented):
    rng = np.random.default_rng(77)
    normal_img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    urgent_img = rng.integers(0, 256, size=(40, 32), dtype=np.uint8)
    layers = parse_model_params(_params())

    results = {}

    def do_normal():
        results["normal"] = _post(
            image=normal_img, urgent=False, client=TestClient(app)
        )
        results["normal_done_at"] = time.monotonic()

    def do_urgent():
        # Start only once the normal job has written a tile, so this really
        # is a mid-flight preemption.
        deadline = time.monotonic() + 5
        while instrumented["snapshot"]() == 0 and time.monotonic() < deadline:
            time.sleep(0.002)
        resp = _post(image=urgent_img, urgent=True, client=TestClient(app))
        results["urgent"] = resp
        results["urgent_done_at"] = time.monotonic()
        results["tiles_at_urgent_done"] = instrumented["snapshot"]()

    tn, errn = _run(do_normal)
    tu, erru = _run(do_urgent)
    tn.join(120)
    tu.join(120)
    assert not errn and not erru

    assert results["urgent"].status_code == 200
    assert results["normal"].status_code == 200
    assert np.array_equal(
        np.load(io.BytesIO(results["urgent"].content)),
        reference_full_inference(urgent_img, layers),
    )
    assert np.array_equal(
        np.load(io.BytesIO(results["normal"].content)),
        reference_full_inference(normal_img, layers),
    )

    # Never more than one image is computing at any moment.
    assert instrumented["max_computing"] == 1
    # The urgent job came back before the normal job finished ...
    assert results["urgent_done_at"] < results["normal_done_at"]
    # ... and the normal job resumed afterwards from its written tiles.
    assert results["tiles_at_urgent_done"] > 0
    assert instrumented["snapshot"]() > results["tiles_at_urgent_done"]
    # Normal 128x128 -> 4x4 tiles per layer; urgent 40x32 -> 2x1 per layer.
    # Every layer-1 tile is a hand-over point; the final layer-2 tile is not
    # (the mask is already complete), so one checkpoint is skipped per job.
    normal_points = 4 * 4 + (4 * 4 - 1)
    urgent_points = 2 * 1 + (2 * 1 - 1)
    assert instrumented["snapshot"]() == normal_points + urgent_points


def test_normal_failure_still_lets_urgent_run(monkeypatch):
    real = main_mod.tiled_segment

    def failing(image_path, layers, feat_path, out_path, tile=512,
                checkpoint=None):
        raise OSError("disk exploded")

    monkeypatch.setattr(main_mod, "tiled_segment", failing)
    first = _post(image=np.zeros((8, 8), np.uint8))
    assert first.status_code == 400

    monkeypatch.setattr(main_mod, "tiled_segment", real)
    urgent = _post(image=np.zeros((8, 8), np.uint8), urgent=True)
    assert urgent.status_code == 200


def test_urgent_failure_does_not_stop_the_normal_job(instrumented):
    """An urgent job that fails mid-flight returns 500; normal continues."""
    rng = np.random.default_rng(3)
    normal_img = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
    urgent_img = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    layers = parse_model_params(_params())
    results = {}
    instrumented_job = main_mod.tiled_segment
    urgent_armed = threading.Event()

    def failing_urgent(image_path, layers_, feat_path, out_path, tile=512,
                       checkpoint=None):
        # Only the urgent image fails, and only once the normal job has
        # written at least one tile.
        if urgent_armed.is_set():
            raise RuntimeError("urgent imager offline")
        return instrumented_job(
            image_path, layers_, feat_path, out_path, tile=tile,
            checkpoint=checkpoint,
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main_mod, "tiled_segment", failing_urgent)
    try:
        def do_normal():
            results["normal"] = _post(
                image=normal_img, urgent=False, client=TestClient(app)
            )

        def do_urgent():
            deadline = time.monotonic() + 5
            while instrumented["snapshot"]() == 0 and time.monotonic() < deadline:
                time.sleep(0.002)
            urgent_armed.set()
            results["urgent"] = _post(
                image=urgent_img, urgent=True, client=TestClient(app)
            )

        tn, errn = _run(do_normal)
        tu, erru = _run(do_urgent)
        tn.join(120)
        tu.join(120)
    finally:
        monkeypatch.undo()
    assert not errn and not erru

    assert results["urgent"].status_code == 500
    assert "inference failed" in results["urgent"].json()["detail"]
    # The normal job was preempted, the urgent job failed, and the normal
    # job still finished correctly from its written tiles.
    assert results["normal"].status_code == 200
    assert np.array_equal(
        np.load(io.BytesIO(results["normal"].content)),
        reference_full_inference(normal_img, layers),
    )
    assert instrumented["max_computing"] == 1


def test_oversized_urgent_never_interrupts_a_running_normal_job(
    instrumented, monkeypatch
):
    rng = np.random.default_rng(4)
    normal_img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    layers = parse_model_params(_params())
    results = {}
    yielded = []

    sched = main_mod._scheduler
    real_checkpoint = sched.checkpoint

    def spy(ticket):
        was_yielded = ticket.yielded
        real_checkpoint(ticket)
        if not was_yielded and ticket.yielded:
            yielded.append(ticket)

    monkeypatch.setattr(sched, "checkpoint", spy)

    def do_normal():
        results["normal"] = _post(
            image=normal_img, urgent=False, client=TestClient(app)
        )

    def do_urgent():
        deadline = time.monotonic() + 5
        while instrumented["snapshot"]() == 0 and time.monotonic() < deadline:
            time.sleep(0.002)
        results["urgent"] = _post(
            image=np.zeros((1025, 4), np.uint8), urgent=True,
            client=TestClient(app),
        )

    tn, errn = _run(do_normal)
    tu, erru = _run(do_urgent)
    tn.join(120)
    tu.join(120)
    assert not errn and not erru
    assert results["urgent"].status_code == 400
    assert results["normal"].status_code == 200
    assert np.array_equal(
        np.load(io.BytesIO(results["normal"].content)),
        reference_full_inference(normal_img, layers),
    )
    # Rejected before acceptance: the normal job was never asked to yield.
    assert yielded == []
    assert instrumented["max_computing"] == 1
