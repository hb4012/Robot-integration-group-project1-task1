from setuptools import find_packages, setup


package_name = "desktop_object_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="student",
    maintainer_email="student@example.com",
    description="Three-class YOLO11 detector for Jetson and ROS 2.",
    license="AGPL-3.0",
    entry_points={
        "console_scripts": [
            "detector_node = desktop_object_detector.detector_node:main",
            "result_logger = desktop_object_detector.result_logger:main",
        ],
    },
)
