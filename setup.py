"""Compatibility shim for editable installs on older pip/setuptools versions."""

from setuptools import find_packages, setup


setup(
    name="ego",
    version="0.1.0",
    description=(
        "Scaffold for egocentric action anticipation, VLM alignment, "
        "and dynamic planning research."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["ego=ego.cli:main"]},
)
