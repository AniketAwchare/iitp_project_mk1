from setuptools import setup, find_packages

setup(
    name="llm-observability",
    version="1.0.0",
    author="Aniket Awchare",
    description="Observability & Reliability Framework for Production LLM Systems — Financial Services",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "llm-observe-serve=llm_observability.core.pipeline:main",
        ]
    },
)
