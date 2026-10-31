"""Read-only scientific model metadata, separate from deployment readiness."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ContractModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def integer_schema_version(cls, value: Any) -> Any:
        # Literal[1] alone also accepts True and 1.0 in Pydantic. A version
        # discriminator must retain the JSON integer type before comparison.
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


class QuantityUnit(_ContractModel):
    quantity: str
    unit: Literal["reduced", "dimensionless"]
    dimension: str
    interpretation: str


class ScientificInvariant(_ContractModel):
    name: str
    definition: str
    applicability: str
    enforcement: Literal["runtime", "reference_tests", "runtime_and_reference_tests"]


class ModelDiagnostic(_ContractModel):
    name: str
    location: str
    interpretation: str


class BackendContract(_ContractModel):
    backend: Literal["numpy", "numba"]
    dependency: str
    optional_extra: str | None
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"
    semantics: str
    silent_fallback: Literal[False] = False


class BackendAvailability(_ContractModel):
    backend: Literal["numpy", "numba"]
    dependency_present: bool
    scope: Literal["catalog_process"] = "catalog_process"
    worker_readiness: Literal["not_checked"] = "not_checked"
    interpretation: str


class RequiredArtifact(_ContractModel):
    name: str
    filename: str
    media_type: str


class UncertaintyCapability(_ContractModel):
    name: str
    availability: Literal["available", "conditional", "unavailable"]
    method: str
    independent_units: str
    assumptions: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)


class ModelVersionPolicy(_ContractModel):
    scope: Literal["declarative_metadata"] = "declarative_metadata"
    behavior_changes_require_new_model_version: Literal[True] = True
    implementation_identity: str
    enforcement: str


class ModelContract(_ContractModel):
    """A fresh contract snapshot; nested JSON schemas are caller-owned copies.

    Freezing this metadata prevents field assignment. It does not prove that
    every implementation obeys the scientific contract or freeze source code.
    """

    schema_version: Literal[1] = 1
    model_version: str
    name: str
    dimensions: Literal[2] = 2
    submission_path: str
    request_schema: dict[str, Any]
    result_schema: dict[str, Any]
    schema_limitations: tuple[str, ...] = Field(min_length=1)
    units: tuple[QuantityUnit, ...] = Field(min_length=1)
    assumptions: tuple[str, ...] = Field(min_length=1)
    invariants: tuple[ScientificInvariant, ...] = Field(min_length=1)
    diagnostics: tuple[ModelDiagnostic, ...] = Field(min_length=1)
    backends: tuple[BackendContract, ...] = Field(min_length=1)
    default_backend: Literal["numpy"] = "numpy"
    required_artifacts: tuple[RequiredArtifact, ...] = Field(min_length=1)
    uncertainty_capabilities: tuple[UncertaintyCapability, ...] = Field(min_length=1)
    version_policy: ModelVersionPolicy
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_declarations(self) -> "ModelContract":
        for records, field in (
            (self.units, "quantity"),
            (self.invariants, "name"),
            (self.diagnostics, "name"),
            (self.backends, "backend"),
            (self.required_artifacts, "name"),
            (self.uncertainty_capabilities, "name"),
        ):
            names = [getattr(record, field) for record in records]
            if len(names) != len(set(names)):
                raise ValueError(f"Contract declarations must have unique {field} values")
        if self.default_backend not in {record.backend for record in self.backends}:
            raise ValueError("The default backend must be declared")
        if "manifest" not in {record.name for record in self.required_artifacts}:
            raise ValueError("A completed model must declare its manifest artifact")
        if any(
            schema.get("type") != "object" for schema in (self.request_schema, self.result_schema)
        ):
            raise ValueError("Request and result schemas must describe objects")
        return self


class ModelCatalog(_ContractModel):
    schema_version: Literal[1] = 1
    models: tuple[ModelContract, ...]
    backend_availability: tuple[BackendAvailability, ...]
