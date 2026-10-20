"""
Configuration management system for lego-agent.
Provides a centralized way to manage configuration settings with variable substitution.
"""

import os
import re
import yaml
from typing import Dict, Any, Optional, Union, Type, TypeVar, Generic, cast
from pathlib import Path
from pydantic import BaseModel, create_model, Field

T = TypeVar("T", bound=BaseModel)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


class VariableSubstitutor:
    """Handles variable substitution in configuration values."""

    # Pattern to match ${VAR} or ${VAR:-default}
    VARIABLE_PATTERN = re.compile(r"\$\{([^}:]+)(?::(-?)([^}]*))?\}")

    @classmethod
    def substitute_variables(
        cls, value: Any, context: Optional[Dict[str, str]] = None
    ) -> Any:
        """Recursively substitute variables in configuration values.

        Args:
            value: The value to process (can be string, dict, list, etc.)
            context: Additional context variables to use for substitution

        Returns:
            The value with variables substituted
        """
        if context is None:
            context = {}

        if isinstance(value, str):
            return cls._substitute_string(value, context)
        elif isinstance(value, dict):
            return {k: cls.substitute_variables(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.substitute_variables(item, context) for item in value]
        else:
            return value

    @classmethod
    def _substitute_string(cls, text: str, context: Dict[str, str]) -> str:
        """Substitute variables in a string.

        Args:
            text: The string to process
            context: Additional context variables

        Returns:
            String with variables substituted
        """

        def replace_match(match):
            var_name = match.group(1).strip()
            # Check if there's a default value specification
            if match.group(2) is not None:  # There's a colon
                hyphen = match.group(2)  # The hyphen (if present)
                default_value = match.group(3)  # The actual default value
                # If there's a hyphen, it means we have :-default syntax
                if hyphen == "-":
                    default_value = default_value if default_value else ""
                else:
                    # Just :default syntax (without hyphen)
                    default_value = hyphen + default_value
            else:
                default_value = ""

            # First check context, then environment variables
            if var_name in context:
                return context[var_name]
            elif var_name in os.environ:
                return os.environ[var_name]
            else:
                return default_value

        return cls.VARIABLE_PATTERN.sub(replace_match, text)

    @classmethod
    def extract_variables(cls, value: Any) -> set:
        """Extract all variable names from a configuration value.

        Args:
            value: The value to scan for variables

        Returns:
            Set of variable names found
        """
        variables = set()

        if isinstance(value, str):
            matches = cls.VARIABLE_PATTERN.findall(value)
            variables.update(var_name.strip() for var_name, _, _ in matches)
        elif isinstance(value, dict):
            for v in value.values():
                variables.update(cls.extract_variables(v))
        elif isinstance(value, list):
            for item in value:
                variables.update(cls.extract_variables(item))

        return variables


class ConfigurationManager:
    """Manages configuration settings for the application."""

    def __init__(self, config_dir: Optional[str] = None, enable_variables: bool = True):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files. Defaults to 'config'.
            enable_variables: Whether to enable variable substitution in configs.
        """
        self.config_dir = config_dir or os.environ.get(
            "LEGO_AGENT_CONFIG_DIR", "config"
        )
        self.enable_variables = enable_variables
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.raw_configs: Dict[str, Dict[str, Any]] = (
            {}
        )  # Store raw configs before substitution
        self._variable_context: Dict[str, str] = {}
        self._load_default_configs()

    def set_variable_context(self, context: Dict[str, str]) -> None:
        """Set additional context variables for substitution.

        Args:
            context: Dictionary of variable name -> value mappings
        """
        self._variable_context = context.copy()
        # Clear processed configs to force re-processing with new context
        self.configs.clear()

    def update_variable_context(self, context: Dict[str, str]) -> None:
        """Update the variable context with additional variables.

        Args:
            context: Dictionary of variable name -> value mappings to add/update
        """
        self._variable_context.update(context)
        # Clear processed configs to force re-processing with new context
        self.configs.clear()

    def get_variable_context(self) -> Dict[str, str]:
        """Get the current variable context.

        Returns:
            Copy of the current variable context
        """
        return self._variable_context.copy()

    def _load_default_configs(self) -> None:
        """Load default configuration files from the config directory."""
        config_path = Path(self.config_dir)
        if not config_path.exists():
            return

        for file_path in config_path.glob("*.yaml"):
            self.load_config(file_path.stem)

    def load_config(self, name: str) -> Dict[str, Any]:
        """Load a configuration file by name.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            The loaded configuration as a dictionary
        """
        # Check if we already have a processed version
        if name in self.configs:
            return self.configs[name]

        file_path = Path(self.config_dir) / f"{name}.yaml"
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                raw_config = yaml.safe_load(f)
                raw_config = raw_config or {}

                # Store raw config
                self.raw_configs[name] = raw_config

                # Process variables if enabled
                if self.enable_variables:
                    processed_config = VariableSubstitutor.substitute_variables(
                        raw_config, self._variable_context
                    )
                else:
                    processed_config = raw_config

                self.configs[name] = processed_config
                print(
                    f"Loaded configuration '{name}' from {file_path}: {processed_config}"
                )
                return self.configs[name]

        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def reload_config(self, name: str) -> Dict[str, Any]:
        """Reload a configuration file, forcing a fresh read from disk.

        Args:
            name: Name of the configuration file

        Returns:
            The reloaded configuration dictionary
        """
        # Remove from caches to force reload
        self.configs.pop(name, None)
        self.raw_configs.pop(name, None)
        return self.load_config(name)

    def get_config(
        self, name: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get a configuration by name, loading it if necessary.

        Args:
            name: Name of the configuration
            default: Default value to return if configuration is not found

        Returns:
            The configuration dictionary
        """
        if name not in self.configs:
            try:
                return self.load_config(name)
            except ConfigurationError:
                if default is not None:
                    return default
                raise ConfigurationError(f"Configuration '{name}' not found.")
        return self.configs[name]

    def get_raw_config(
        self, name: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get the raw configuration before variable substitution.

        Args:
            name: Name of the configuration
            default: Default value to return if configuration is not found

        Returns:
            The raw configuration dictionary
        """
        if name not in self.raw_configs:
            try:
                return self.load_config(name)
            except ConfigurationError:
                if default is not None:
                    return default
                raise ConfigurationError(f"Raw configuration '{name}' not found.")
        return self.raw_configs[name]

    def get_config_value(self, name: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value.

        Args:
            name: Name of the configuration
            key: Configuration key (supports dot notation for nested configs)
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        config = self.get_config(name)
        keys = key.split(".")

        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_typed_config(self, name: str, model_class: Type[T]) -> T:
        """Get a typed configuration using a Pydantic model.

        Args:
            name: Name of the configuration
            model_class: Pydantic model class to validate against

        Returns:
            An instance of the model with configuration values
        """
        config = self.get_config(name)
        return model_class.model_validate(config)

    def get_config_variables(self, name: str) -> set:
        """Get all variable names used in a configuration.

        Args:
            name: Name of the configuration

        Returns:
            Set of variable names found in the configuration
        """
        raw_config = self.get_raw_config(name)
        return VariableSubstitutor.extract_variables(raw_config)

    def validate_config_variables(self, name: str) -> Dict[str, bool]:
        """Validate that all variables in a configuration can be resolved.

        Args:
            name: Name of the configuration

        Returns:
            Dictionary mapping variable names to whether they can be resolved
        """
        variables = self.get_config_variables(name)
        validation_result = {}

        for var_name in variables:
            can_resolve = var_name in self._variable_context or var_name in os.environ
            validation_result[var_name] = can_resolve

        return validation_result


def create_config_manager(
    config_dir: Optional[str] = None, enable_variables: bool = True
) -> ConfigurationManager:
    """Create a new configuration manager instance.

    This factory function allows creating custom configuration managers
    for testing or specialized use cases.

    Args:
        config_dir: Optional custom configuration directory
        enable_variables: Whether to enable variable substitution

    Returns:
        A new ConfigurationManager instance
    """
    return ConfigurationManager(config_dir, enable_variables)


# Global configuration manager instance
config_manager = create_config_manager()
