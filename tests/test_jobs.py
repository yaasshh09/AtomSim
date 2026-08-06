import pytest

from atomsim.server.jobs import JobStatus, JobStore


def test_create_and_get():
    store = JobStore()
    job = store.create()
    assert job.status is JobStatus.PENDING
    assert store.get(job.id) is job
    assert store.get("nope") is None


def test_run_success_sets_done_result_and_full_progress():
    store = JobStore()
    job = store.create()
    seen: list[float] = []

    def work(progress):
        progress(0.5)
        seen.append(store.get(job.id).progress)
        return "payload"

    store.run(job.id, work)
    assert seen == [0.5]  # progress visible to observers mid-run
    assert job.status is JobStatus.DONE
    assert job.result == "payload"
    assert job.progress == pytest.approx(1.0)


def test_run_failure_sets_error_status_and_message():
    store = JobStore()
    job = store.create()

    def bad(progress):
        raise ValueError("boom")

    store.run(job.id, bad)
    assert job.status is JobStatus.ERROR
    assert "ValueError" in job.error and "boom" in job.error
    assert job.result is None


def test_progress_is_clamped_to_unit_interval():
    store = JobStore()
    job = store.create()

    def work(progress):
        progress(7.0)
        assert job.progress == 1.0
        progress(-3.0)
        assert job.progress == 0.0
        return None

    store.run(job.id, work)


def test_run_unknown_job_raises():
    store = JobStore()
    with pytest.raises(KeyError):
        store.run("nope", lambda progress: None)


def _finish(store, job):
    store.run(job.id, lambda progress: "payload")
    return job


def test_finished_jobs_are_evicted_oldest_first_once_the_cap_is_passed():
    store = JobStore(max_jobs=3)
    done = [_finish(store, store.create()) for _ in range(3)]
    assert len(store) == 3

    fourth = store.create()

    # The oldest finished job went; everything newer survived.
    assert store.get(done[0].id) is None
    assert [store.get(j.id) for j in done[1:]] == done[1:]
    assert store.get(fourth.id) is fourth
    assert len(store) == 3


def test_unfinished_jobs_are_never_evicted_even_over_the_cap():
    """A vanished RUNNING job would strand both its worker and its watcher."""
    store = JobStore(max_jobs=2)
    pending = [store.create() for _ in range(5)]

    # Nothing has finished, so the cap yields rather than dropping live work.
    assert len(store) == 5
    assert all(store.get(job.id) is job for job in pending)

    # Once they finish, the next create collects the backlog down to the cap.
    for job in pending:
        store.run(job.id, lambda progress: None)
    newest = store.create()
    assert len(store) == 2
    assert store.get(newest.id) is newest
    assert store.get(pending[0].id) is None


def test_eviction_notifies_so_side_tables_can_be_purged():
    evicted: list[str] = []
    store = JobStore(max_jobs=1, on_evict=evicted.append)
    first = _finish(store, store.create())
    store.create()
    assert evicted == [first.id]


def test_no_eviction_callback_when_nothing_is_dropped():
    evicted: list[str] = []
    store = JobStore(max_jobs=8, on_evict=evicted.append)
    for _ in range(4):
        _finish(store, store.create())
    assert evicted == []
