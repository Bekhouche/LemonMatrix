from urllib.parse import urlparse

import pytest

from lemonmatrix.discover import expand_subnet, scan_localhost


def _port_of(url: str) -> int:
    return urlparse(url).port


def test_scan_localhost_finds_the_fake_server(fake_lemonade):
    port = _port_of(fake_lemonade)
    found = scan_localhost(ports=[port], timeout=1.0)
    assert found == [("127.0.0.1", port)]


def test_scan_localhost_ignores_closed_ports(fake_lemonade):
    # Port 1 is privileged/unassigned and should not be listening in test envs.
    found = scan_localhost(ports=[1], timeout=0.3)
    assert found == []


def test_expand_subnet_counts_usable_hosts():
    hosts = expand_subnet("192.168.1.0/30")
    # A /30 has 4 addresses, 2 usable host addresses.
    assert len(hosts) == 2


def test_expand_subnet_rejects_ranges_over_the_limit():
    with pytest.raises(ValueError):
        expand_subnet("10.0.0.0/8")
