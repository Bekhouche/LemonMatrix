"""Finds live Lemonade instances by probing candidate host:port pairs.

Lemonade has no announce/broadcast mechanism (no mDNS/SSDP in its docs), so
there is no real way to discover instances elsewhere on a LAN without being
told where to look. What IS reliable: the unversioned GET /live liveness
probe (https://lemonade-server.ai/docs/api/lemonade/) and Lemonade's own
documented port history/quick-select list, which makes localhost scanning
cheap and accurate. Subnet scanning is offered as an explicit opt-in for the
IDEA.md "fleet of LAN machines" scenario, since probing hosts you don't own
is a different kind of action than checking your own loopback interface.
"""

from __future__ import annotations

import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# 13305 is the current default (since Lemonade v10.1); 8000 was the prior
# default; the rest are Lemonade's documented quick-select alternates.
QUICK_SELECT_PORTS = [13305, 8000, 8020, 8040, 8060, 8080, 9000, 11434]

MAX_SUBNET_HOSTS = 1024


def _probe(host: str, port: int, timeout: float) -> tuple[str, int] | None:
    url = f"http://{host}:{port}/live"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.ok:
            return host, port
    except requests.RequestException:
        return None
    return None


def scan(hosts: list[str], ports: list[int] = QUICK_SELECT_PORTS, timeout: float = 0.5) -> list[tuple[str, int]]:
    """Probe every (host, port) pair concurrently; return the ones that answered /live."""
    found: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=min(64, max(1, len(hosts) * len(ports)))) as pool:
        futures = [pool.submit(_probe, host, port, timeout) for host in hosts for port in ports]
        for future in as_completed(futures):
            result = future.result()
            if result:
                found.append(result)
    return sorted(found)


def scan_localhost(ports: list[int] = QUICK_SELECT_PORTS, timeout: float = 0.5) -> list[tuple[str, int]]:
    return scan(["127.0.0.1"], ports=ports, timeout=timeout)


def expand_subnet(cidr: str) -> list[str]:
    network = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(ip) for ip in network.hosts()]
    if len(hosts) > MAX_SUBNET_HOSTS:
        raise ValueError(
            f"{cidr} has {len(hosts)} hosts, over the {MAX_SUBNET_HOSTS} scan limit. Use a narrower range."
        )
    return hosts
