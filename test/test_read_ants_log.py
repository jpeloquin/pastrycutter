from pathlib import Path

from pastrycutter.ants import parse_ants_log

DIR_FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_ants_log_CC():
    pth_log = DIR_FIXTURES / "antsRegistration_log_2level_CC.log"
    expected = {
        "setup": {
            "match_histograms": True,
            "winsorization": {"enabled": True, "quantiles": (0.0, 1.0)},
        },
        "stages": [
            {
                "metrics": [
                    {
                        "metric": "CC",
                        "weight": 1.0,
                        "radius": 3,
                        "filter_gradient": False,
                    }
                ],
                "sampling": None,
                "levels": 2,
                "shrink_factors": ((2, 2, 2), (1, 1, 1)),
                "smoothing": (0, 0),
                "convergence_threshold": 1e-6,
                "convergence_window": 10,
                "max_iterations": (200, 200),
            }
        ],
    }
    actual = parse_ants_log(pth_log)
    assert actual == expected


def test_parse_ants_log_MI():
    pth_log = DIR_FIXTURES / "antsRegistration_log_2level_MI.log"
    expected = {
        "setup": {
            "match_histograms": False,
            "winsorization": {"enabled": True, "quantiles": (0.0, 1.0)},
        },
        "stages": [
            {
                "metrics": [
                    {
                        "metric": "MI",
                        "weight": 1.0,
                        "bins": 32,
                        "filter_gradient": False,
                    }
                ],
                "sampling": None,
                "levels": 2,
                "shrink_factors": ((2, 2, 2), (1, 1, 1)),
                "smoothing": (2, 1),
                "convergence_threshold": 1e-6,
                "convergence_window": 10,
                "max_iterations": (200, 200),
            }
        ],
    }
    actual = parse_ants_log(pth_log)
    assert actual == expected


def test_parse_ants_log_MSQ():
    pth_log = DIR_FIXTURES / "antsRegistration_log_2level_MSQ.log"
    expected = {
        "setup": {
            "match_histograms": True,
            "winsorization": {"enabled": True, "quantiles": (0.0765803, 0.92342)},
        },
        "stages": [
            {
                "metrics": [
                    {
                        "metric": "MSQ",
                        "weight": 1.0,
                        "filter_gradient": True,
                    }
                ],
                "sampling": None,
                "levels": 2,
                "shrink_factors": ((2, 2, 1), (1, 1, 1)),
                "smoothing": (0, 0),
                "convergence_threshold": 1e-6,
                "convergence_window": 10,
                "max_iterations": (200, 200),
            }
        ],
    }
    actual = parse_ants_log(pth_log)
    assert actual == expected


# Make next test target GC, but also something else (like more than one stage, or
# winsorization disabled)


if __name__ == "__main__":
    test_parse_ants_log_CC()
