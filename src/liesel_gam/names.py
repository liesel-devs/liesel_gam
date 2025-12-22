from dataclasses import dataclass, field


@dataclass
class NameManager:
    """
    Creates unique names.
    """

    prefix: str = ""
    """Added to names."""

    created_names: dict[str, int] = field(default_factory=dict)
    """Dictionary created names."""

    def create(self, name: str, apply_prefix: bool = True, lazy: bool = True) -> str:
        """
        Appends a counter to the given name for uniqueness.
        There is an individual counter for each name.

        If a prefix was passed to the builder on init, the prefix is applied to the
        name.
        """
        if apply_prefix:
            name = self.prefix + name

        i = self.created_names.get(name, 0)

        if i > 0 and lazy:
            name_indexed = name + str(i)
        else:
            name_indexed = name

        self.created_names[name] = i + 1

        return name_indexed

    def fname(self, f: str, of: str) -> str:
        """Creates a function name ``{f}({of})``."""
        return self.create(f"{f}({of})")

    def param(
        self,
        param_name: str,
        term_name: str = "",
    ) -> str:
        """Creates a parameter name ``${param_name}_{term_name}$``."""
        param_name = param_name.replace("$", "")
        if term_name:
            term_name = term_name.replace("$", "")
            param_name = f"${param_name}" + "_{" + f"{term_name}" + "}$"
            # apply_prefix false, because the assumption is that any prefix will be
            # present in the term name already
            return self.create(param_name, apply_prefix=False)
        else:
            param_name = f"${param_name}$"
            return self.create(param_name, apply_prefix=True)

    def beta(self, term_name: str = "") -> str:
        """Creates a beta parameter name ``$\\beta_{term_name}$``."""
        return self.param(term_name=term_name, param_name="\\beta")

    def tau(self, term_name: str = "") -> str:
        """Creates a tau parameter name ``$\\tau_{term_name}$``."""
        return self.param(term_name=term_name, param_name="\\tau")

    def tau2(self, term_name: str = "") -> str:
        """Creates a tau2 parameter name ``$\\tau^2_{term_name}$``."""
        return self.param(term_name=term_name, param_name="\\tau^2")
