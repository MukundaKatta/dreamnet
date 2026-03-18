"""Tests for Dreamnet."""
from src.core import Dreamnet
def test_init(): assert Dreamnet().get_stats()["ops"] == 0
def test_op(): c = Dreamnet(); c.search(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Dreamnet(); [c.search() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Dreamnet(); c.search(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Dreamnet(); r = c.search(); assert r["service"] == "dreamnet"
