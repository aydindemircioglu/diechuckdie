
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='diechuckdie',
    version='0.1',
    description='Simple DICOM Helpers',
    long_description='Some simple classes to help with DICOM and Radiomics.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: DICOM :: Radiomics',
    ],
    keywords='DICOM radiomics dce mri',
    url='http://github.com/aydindemircioglu/diechuckdie',
    author='Aydin Demircioglu',
    author_email='diechuckdie@cloned.de',
    license='MIT',
    packages=['diechuckdie'],
    install_requires=[
        'scipy',
    ],
    scripts=['bin/diechuckdie'],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False)
