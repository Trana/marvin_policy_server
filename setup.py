from setuptools import find_packages, setup

package_name = 'marvin_policy_server'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name, package_name + '.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/marvin_policy_server.launch.py']),        
        ('share/' + package_name + '/policy', ['policy/policy.pt']),
        ('share/' + package_name + '/policy', ['policy/env.yaml']),
        ('share/' + package_name + '/policy', ['policy/velocity_estimator.pt']),
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
            'marvin_delay_test = marvin_policy_server.delay_test_node:main',
            'joy_command_recorder = marvin_policy_server.joy_command_recorder_node:main',
            'joy_command_replay = marvin_policy_server.joy_command_replay_node:main',
        ],
    },
)
