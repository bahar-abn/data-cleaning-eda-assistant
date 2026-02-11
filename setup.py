"""
فایل نصب پکیج
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="data-cleaning-eda-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="دستیار هوشمند پاکسازی داده و تحلیل اکتشافی",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/data-cleaning-eda-assistant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eda-assistant=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)