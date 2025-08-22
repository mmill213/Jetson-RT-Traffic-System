from setuptools import setup

package_name = 'videopub'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy'
        ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@email.com',
    description='Python video publisher node for ROS 2',
    license='MIT',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_publisher = videopub.video_publisher:main',
        ],
    },
)
