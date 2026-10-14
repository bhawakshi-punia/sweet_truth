from sweet_truth.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_limit_resets_as_the_window_moves_on():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60, clock=clock)

    assert limiter.retry_after("a") == 0
    clock.now = 10
    assert limiter.retry_after("a") == 0
    clock.now = 20
    assert limiter.retry_after("a") == 40  # the first call expires at 60 s
    assert limiter.retry_after("b") == 0  # other visitors have their own allowance

    clock.now = 60.5
    assert limiter.retry_after("a") == 0
