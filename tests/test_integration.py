"""Integration tests for Dreamnet."""
from src.core import Dreamnet

class TestDreamnet:
    def setup_method(self):
        self.c = Dreamnet()
    def test_10_ops(self):
        for i in range(10): self.c.search(i=i)
        assert self.c.get_stats()["ops"] == 10
    def test_service_name(self):
        assert self.c.search()["service"] == "dreamnet"
    def test_different_inputs(self):
        self.c.search(type="a"); self.c.search(type="b")
        assert self.c.get_stats()["ops"] == 2
    def test_config(self):
        c = Dreamnet(config={"debug": True})
        assert c.config["debug"] is True
    def test_empty_call(self):
        assert self.c.search()["ok"] is True
    def test_large_batch(self):
        for _ in range(100): self.c.search()
        assert self.c.get_stats()["ops"] == 100
