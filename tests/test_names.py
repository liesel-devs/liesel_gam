from liesel_gam.names import NameManager


class TestNameManager:
    def test_param(self):
        nm = NameManager()
        n1 = nm.param("test")
        assert n1 == "$test$"

        n2 = nm.param("test")
        assert n2 == "$test$1"

        n3 = nm.param("test", "a")
        assert n3 == "$test_{a}$"

        n4 = nm.param("test", "a")
        assert n4 == "$test_{a}$1"

    def test_tau2(self):
        nm = NameManager()
        n1 = nm.tau2()
        n2 = nm.tau2()
        assert n1 == "$\\tau^2$"
        assert n2 == "$\\tau^2$1"
