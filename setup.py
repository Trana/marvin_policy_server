from setuptools import setup

package_name = 'marvin_policy_server'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/marvin_policy_server.launch.py']),
        ('share/' + package_name + '/policy', ['policy/marvin_policy.pt']),
        ('share/' + package_name + '/policy', ['policy/policy.pt']),
        ('share/' + package_name + '/policy', ['policy/marvin_env.yaml']),
        ('share/' + package_name + '/policy', ['policy/env.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Policy server',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            # LEFT = name you run, RIGHT = package.module:function
            'marvin_policy_server = marvin_policy_server.marvin_policy_server:main',
        ],
    },
)
