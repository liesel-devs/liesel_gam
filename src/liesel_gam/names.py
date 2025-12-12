from dataclasses import dataclass, field


@dataclass
class NameManager:
    prefix: str = ""
    created_names: dict[str, int] = field(default_factory=dict)

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
        return self.create(f"{f}({of})")

    def param(
        self,
        param_name: str,
        term_name: str = "",
    ) -> str:
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
        return self.param(term_name=term_name, param_name="\\beta")

    def tau(self, term_name: str = "") -> str:
        return self.param(term_name=term_name, param_name="\\tau")

    def tau2(self, term_name: str = "") -> str:
        return self.param(term_name=term_name, param_name="\\tau^2")
