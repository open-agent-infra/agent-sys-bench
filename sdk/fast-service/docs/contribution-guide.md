# Contribution Guide

## Before submit your code

### Add test if necessary

This project use ```pytest``` to run the test files in package [test](../test). The github actions will automatically run all the tests.

### Format your code

Codes in this project shall be formatted. Please follow the following steps to format your code, otherwise the github actions of your PR will fail.

1. Execute ```pip install black``` to install black in your environment.
2. Execute ```black .``` in your project dir to format the code.